"""RAG evaluation with Evidently AI: LLM-as-judge quality checks on the real
chatbot's answers, on top of evals/rag_eval.py's retrieval metrics.

For each golden question it records the chunks Milvus retrieves, the
chatbot's actual answer (POST /chat on a running stack), Precision@K /
Recall@K and both latencies, then scores every row with Evidently's
descriptors - an LLM (gpt-4o-mini) acting as the judge:

- Context quality  - is the retrieved context enough to answer the question?
- Faithfulness     - is the answer grounded in that context (no hallucination)?
- Completeness     - does the answer cover everything the question asks? (a custom judge)
- Declined         - did the chatbot refuse to answer?
- PII in answer    - does the answer contain actual personal data? (a custom judge)
- Answer length    - characters

Output: an interactive evals/evidently_report.html, the scored rows with each
judge's verdict in evals/evidently_rows.csv, and a summary printed and appended
to evals/report.md.

    make eval-evidently        # or, from the host:
    VECTOR_DB_URI=http://localhost:19530 uv run --group eval python -m evals.evidently_eval

Needs the stack running (static docs ingested) and OPENAI_API_KEY - each row
makes ~5 judge calls, so a run costs a few cents. Evidently and pandas are
imported lazily: they are in the `eval` dependency group only.
"""

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from app.config import settings
from evals.rag_eval import CHAT_REQUEST_SPACING_SECONDS, connect_vector_repository, load_golden_set

EVALS_DIR = Path(__file__).resolve().parent
JUDGE_MODEL = "gpt-4o-mini"

# Judge columns: alias -> the category that counts as a pass.
JUDGED = {
    "Context quality": "VALID",
    "Faithfulness": "FAITHFUL",
    "Completeness": "COMPLETE",
    "Declined": "OK",  # DeclineLLMEval: "DECLINE" when the chatbot refused
    "PII in answer": "OK",  # PIILLMEval: "PII" when it found personal data
}

Ask = Callable[[str], tuple[str, float]]


def collect_rows(retriever, golden_set: list[dict], k: int, ask: Ask) -> list[dict]:
    """One row per golden question: retrieved context, the chatbot's answer,
    retrieval metrics and latencies. `ask(question) -> (answer, latency_ms)`."""
    rows = []
    for case in golden_set:
        relevant = set(case["relevant_sources"])

        started = time.perf_counter()
        documents = retriever.search(case["question"], k=k)
        retrieval_ms = (time.perf_counter() - started) * 1000

        sources = [document.metadata.get("source", "unknown") for document in documents]
        answer, chat_ms = ask(case["question"])

        rows.append(
            {
                "question": case["question"],
                "context": "\n\n".join(document.page_content for document in documents),
                "answer": answer,
                "retrieved_sources": ", ".join(sources),
                f"precision_at_{k}": sum(source in relevant for source in sources) / len(sources) if sources else 0.0,
                f"recall_at_{k}": len(relevant & set(sources)) / len(relevant) if relevant else 0.0,
                "retrieval_ms": round(retrieval_ms, 1),
                "chat_ms": round(chat_ms, 1),
            }
        )
    return rows


def chat_asker(api_url: str) -> tuple[Ask, httpx.Client]:
    """Asks the running chatbot, one session, spaced under its rate limit."""
    client = httpx.Client(base_url=api_url, timeout=90)
    client.get("/health").raise_for_status()

    def ask(question: str) -> tuple[str, float]:
        # A fresh session per question, so answers don't depend on earlier ones:
        # GET / keeps an existing cookie, so drop it first.
        client.cookies.clear()
        client.get("/")
        started = time.perf_counter()
        response = client.post("/chat", json={"question": question})
        response.raise_for_status()
        elapsed = (time.perf_counter() - started) * 1000
        time.sleep(CHAT_REQUEST_SPACING_SECONDS)
        return response.json()["answer"], elapsed

    return ask, client


# Evidently's built-in PIILLMEval flags an answer that merely *names* a kind of
# personal data ("you'll need your name and car number"); this judge only fails
# actual values, and treats the company's public contacts as not personal.
PII_CRITERIA = """
The text is a parking chatbot's answer. Classify it as PII only if it contains an actual
value of a specific person's personal data: a real name, a car registration (licence plate)
number, a personal phone number or email address, a home address, an ID or payment card number.

These are NOT PII:
- naming a kind of personal data without a value, e.g. "you will need your first name, surname
  and car registration number";
- the car park operator's public business contacts (company phone, info@ email, office address);
- placeholders or redaction markers such as <PERSON> or [REDACTED].
""".strip()


def pii_judge():
    """Evidently LLM judge for actual personal data in the answer (see PII_CRITERIA)."""
    from evidently.descriptors import LLMEval
    from evidently.llm.models import LLMMessage
    from evidently.llm.templates import BinaryClassificationPromptTemplate, Uncertainty

    template = BinaryClassificationPromptTemplate(
        criteria=PII_CRITERIA,
        target_category="PII",
        non_target_category="OK",
        uncertainty=Uncertainty.NON_TARGET,  # unsure means no value was found
        include_reasoning=True,
        pre_messages=[LLMMessage.system("You are a judge which evaluates text.")],
    )
    return LLMEval("answer", provider="openai", model=JUDGE_MODEL, template=template, alias="PII in answer")


# Evidently's built-in CompletenessLLMEval asks whether the answer repeats
# *everything* in the retrieved chunks, so "10 km/h" fails for not also listing
# EV charger penalties. This judge asks whether it answers the question fully.
COMPLETENESS_CRITERIA = """
The text is a parking chatbot's answer to the QUESTION below. Judge it against the SOURCE,
the documents retrieved for that question.

An answer is COMPLETE if it gives every fact in the SOURCE that the QUESTION asks for:
- all items when the question asks for a list, steps or options;
- the conditions and exceptions that change the answer (who a rule applies to, a car park
  where a limit differs).
Facts in the SOURCE that the QUESTION doesn't ask about are not needed - leaving them out is
correct, not incomplete. Extra correct details do not make an answer incomplete.

An answer is INCOMPLETE only if it leaves out such a question-relevant fact, item or condition.

-----question_starts-----
{question}
-----question_ends-----

-----source_starts-----
{context}
-----source_finishes-----
""".strip()


def completeness_judge():
    """Evidently LLM judge: does the answer cover everything the question asks?
    (see COMPLETENESS_CRITERIA)."""
    from evidently.descriptors import LLMEval
    from evidently.llm.models import LLMMessage
    from evidently.llm.templates import BinaryClassificationPromptTemplate, Uncertainty

    template = BinaryClassificationPromptTemplate(
        criteria=COMPLETENESS_CRITERIA,
        target_category="INCOMPLETE",
        non_target_category="COMPLETE",
        uncertainty=Uncertainty.UNKNOWN,
        include_reasoning=True,
        pre_messages=[LLMMessage.system("You are an impartial expert evaluator of answer completeness.")],
    )
    return LLMEval(
        "answer",
        provider="openai",
        model=JUDGE_MODEL,
        template=template,
        additional_columns={"question": "question", "context": "context"},
        alias="Completeness",
    )


def descriptors():
    """Evidently's per-row checks; the LLM judges use JUDGE_MODEL via OPENAI_API_KEY."""
    from evidently.descriptors import (
        ContextQualityLLMEval,
        DeclineLLMEval,
        FaithfulnessLLMEval,
        TextLength,
    )

    return [
        ContextQualityLLMEval("context", question="question", model=JUDGE_MODEL, alias="Context quality"),
        FaithfulnessLLMEval("answer", context="context", model=JUDGE_MODEL, alias="Faithfulness"),
        completeness_judge(),
        DeclineLLMEval("answer", model=JUDGE_MODEL, alias="Declined"),
        pii_judge(),
        TextLength("answer", alias="Answer length"),
    ]


def summarize(scored_rows: list[dict]) -> dict[str, float]:
    """Share of rows passing each judge, from rows that carry the judge columns."""
    summary = {}
    for alias, passing in JUDGED.items():
        values = [row.get(alias) for row in scored_rows if row.get(alias) is not None]
        summary[alias] = sum(value == passing for value in values) / len(values) if values else float("nan")
    return summary


def render_summary(summary: dict[str, float], rows: int, k: int) -> str:
    lines = [
        "",
        "## Evidently AI: LLM-as-judge answer quality",
        "",
        f"{rows} golden questions, real chatbot answers, judged by `{JUDGE_MODEL}` with Evidently "
        "descriptors - full interactive report: [`evidently_report.html`](evidently_report.html).",
        "",
        "| Check | Passing | Pass means |",
        "| --- | --- | --- |",
    ]
    meaning = {
        "Context quality": f"the top-{k} retrieved chunks are enough to answer",
        "Faithfulness": "the answer is grounded in the retrieved context",
        "Completeness": "the answer gives every fact and condition the question asks for",
        "Declined": "the chatbot answered instead of refusing",
        "PII in answer": "no actual personal data (a name, plate, personal contact) in the answer",
    }
    for alias, share in summary.items():
        lines.append(f"| {alias} | {share:.0%} | {meaning[alias]} |")
    lines += [
        "",
        "How to read it - check the per-row reasoning in `evidently_rows.csv` before trusting a number:",
        "",
        f"- The judges compare each answer with the top-{k} chunks retrieved for the *question as asked*. "
        "The chatbot searches with its own wording and also answers from the database tools (prices, hours, "
        "availability), so a correct answer can be marked unfaithful or incomplete.",
        '- LLM judges are noisy: a plain "no" can be read as a refusal, and verdicts vary a little between runs.',
        "- Completeness and PII use custom judges (`COMPLETENESS_CRITERIA`, `PII_CRITERIA`). Evidently's "
        "built-in Completeness wants the answer to repeat every retrieved chunk (13% here, failing "
        '"10 km/h" for not listing EV charger penalties); its PII judge flagged an answer for saying a '
        "booking needs a name and car number.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG evaluation with Evidently AI")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--k", type=int, default=settings.retriever_k)
    args = parser.parse_args()

    import pandas as pd
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import TextEvals

    # Evidently's OpenAI judge reads the key from the environment.
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key.get_secret_value())

    golden_set = load_golden_set()
    repository = connect_vector_repository()
    ask, client = chat_asker(args.api_url)
    try:
        repository.load_collection()
        rows = collect_rows(repository, golden_set, args.k, ask)
    finally:
        client.close()
        repository.close()

    dataset = Dataset.from_pandas(
        pd.DataFrame(rows),
        data_definition=DataDefinition(
            text_columns=["question", "context", "answer"],
            numerical_columns=[f"precision_at_{args.k}", f"recall_at_{args.k}", "retrieval_ms", "chat_ms"],
        ),
        descriptors=descriptors(),
    )
    snapshot = Report([TextEvals()]).run(dataset, None)
    html_path = EVALS_DIR / "evidently_report.html"
    snapshot.save_html(str(html_path))

    scored = dataset.as_dataframe()
    scored.to_csv(EVALS_DIR / "evidently_rows.csv", index=False)  # every row with its judge verdicts
    summary = summarize(scored.to_dict("records"))
    section = render_summary(summary, len(rows), args.k)
    print(section)
    print(f"Interactive report written to {html_path}")

    report_path = EVALS_DIR / "report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "\n## Evidently AI:"
    report_path.write_text(existing.split(marker)[0].rstrip() + "\n" + section, encoding="utf-8")


if __name__ == "__main__":
    main()
