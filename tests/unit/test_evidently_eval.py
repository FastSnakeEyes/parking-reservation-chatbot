"""evals/evidently_eval.py's data collection and summary - the parts that
don't need Evidently or pandas (they're in the `eval` group, not CI's)."""

import math

from evals.evidently_eval import collect_rows, render_summary, summarize
from langchain_core.documents import Document


class FakeRetriever:
    def search(self, query, k):
        return [
            Document(page_content="Free cancellation up to 2 hours before.", metadata={"source": "booking_process.md"}),
            Document(page_content="VilniusPark runs five car parks.", metadata={"source": "general_info.md"}),
        ][:k]


def test_collect_rows_records_context_answer_and_metrics():
    golden = [{"question": "Can I cancel for free?", "relevant_sources": ["booking_process.md"]}]

    rows = collect_rows(FakeRetriever(), golden, k=2, ask=lambda question: (f"Yes. ({question})", 1234.5))

    row = rows[0]
    assert "Free cancellation up to 2 hours before." in row["context"]
    assert row["answer"] == "Yes. (Can I cancel for free?)"
    assert row["retrieved_sources"] == "booking_process.md, general_info.md"
    assert (row["precision_at_2"], row["recall_at_2"]) == (0.5, 1.0)
    assert row["chat_ms"] == 1234.5


def test_summarize_counts_each_judges_passing_category():
    scored = [
        {"Context quality": "VALID", "Faithfulness": "FAITHFUL", "Completeness": "INCOMPLETE", "Declined": "OK"},
        {"Context quality": "INVALID", "Faithfulness": "FAITHFUL", "Completeness": "COMPLETE", "Declined": "DECLINE"},
    ]

    summary = summarize(scored)

    assert summary["Context quality"] == 0.5
    assert summary["Faithfulness"] == 1.0
    assert summary["Completeness"] == 0.5
    assert summary["Declined"] == 0.5
    assert math.isnan(summary["PII in answer"])  # no verdicts for it in these rows


def test_render_summary_is_a_markdown_table():
    section = render_summary({"Context quality": 0.8, "Faithfulness": 1.0}, rows=15, k=4)

    assert "## Evidently AI" in section
    assert "| Context quality | 80% |" in section
    assert "| Faithfulness | 100% |" in section
