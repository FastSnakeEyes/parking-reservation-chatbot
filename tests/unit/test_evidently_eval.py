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


def test_collect_rows_has_no_retrieval_metrics_for_a_question_to_decline():
    golden = [{"question": "Book me a taxi", "category": "out_of_scope", "expected": "decline", "relevant_sources": []}]

    rows = collect_rows(FakeRetriever(), golden, k=2, ask=lambda question: ("I can only help with parking.", 1.0))

    assert (rows[0]["expected"], rows[0]["category"]) == ("decline", "out_of_scope")
    assert rows[0]["precision_at_2"] is None and rows[0]["recall_at_2"] is None


def test_summarize_scores_each_check_on_the_questions_it_applies_to():
    answered = {
        "expected": "answer",
        "category": "single_fact",
        "Context quality": "VALID",
        "Faithfulness": "FAITHFUL",
        "Completeness": "INCOMPLETE",
        "Declined": "OK",
        "PII in answer": "OK",
    }
    declined = {
        "expected": "decline",
        "category": "out_of_scope",
        "Context quality": "INVALID",  # nothing to retrieve - not counted for a question to decline
        "Faithfulness": "UNFAITHFUL",
        "Completeness": "INCOMPLETE",
        "Declined": "DECLINE",  # the right behaviour here
        "PII in answer": "OK",
    }

    summary = summarize([answered, declined])

    assert summary["checks"]["Context quality"] == 1.0
    assert summary["checks"]["Completeness"] == 0.0
    assert summary["checks"]["Answered or declined correctly"] == 1.0
    assert summary["categories"] == {"single_fact": (0.0, 1), "out_of_scope": (1.0, 1)}
    assert summary["overall"] == 0.5


def test_answering_a_question_to_decline_fails_the_behaviour_check():
    row = {"expected": "decline", "category": "injection", "Declined": "OK", "PII in answer": "OK"}

    summary = summarize([row])

    assert summary["checks"]["Answered or declined correctly"] == 0.0
    assert math.isnan(summary["checks"]["Faithfulness"])  # not applicable


def test_render_summary_is_a_markdown_table():
    summary = {
        "checks": {"Context quality": 0.8, "Faithfulness": 1.0},
        "categories": {"paraphrase": (0.6, 5)},
        "overall": 0.75,
    }

    section = render_summary(summary, rows=30, k=3)

    assert "## Evidently AI" in section
    assert "**Overall: 75%**" in section
    assert "| Context quality | 80% |" in section
    assert "| paraphrase | 5 | 60% |" in section
