"""The RAG evaluation's metrics (evals/rag_eval.py), on a fake retriever."""

import pytest
from evals.rag_eval import answerable, evaluate_retrieval, recommend_k, render_report, sweep_k
from langchain_core.documents import Document


class FakeRetriever:
    def __init__(self, results: dict[str, list[str]]):
        self.results = results  # question -> sources of the retrieved chunks, in rank order

    def search(self, query, k):
        return [Document(page_content="chunk", metadata={"source": source}) for source in self.results[query][:k]]


def test_precision_counts_relevant_chunks_and_recall_relevant_documents():
    golden = [
        {"question": "q1", "relevant_sources": ["rules.md"]},
        {"question": "q2", "relevant_sources": ["a.md", "b.md"]},
    ]
    retriever = FakeRetriever(
        {
            "q1": ["rules.md", "rules.md", "rules.md", "other.md"],  # 3 of 4 chunks relevant
            "q2": ["a.md", "other.md", "other.md", "other.md"],  # finds only a.md
        }
    )

    result = evaluate_retrieval(retriever, golden, k=4)

    assert [row["precision"] for row in result["per_query"]] == [0.75, 0.25]
    assert [row["recall"] for row in result["per_query"]] == [1.0, 0.5]
    assert result["mean_precision"] == 0.5
    assert result["mean_recall"] == 0.75
    # Both have a relevant chunk at rank 1.
    assert result["mrr"] == 1.0
    assert result["precision_at_1"] == 1.0


def test_mrr_rewards_ranking_the_relevant_chunk_first():
    golden = [
        {"question": "first", "relevant_sources": ["a.md"]},
        {"question": "third", "relevant_sources": ["a.md"]},
        {"question": "missing", "relevant_sources": ["a.md"]},
    ]
    retriever = FakeRetriever(
        {
            "first": ["a.md", "x.md", "y.md"],
            "third": ["x.md", "y.md", "a.md"],
            "missing": ["x.md", "y.md", "z.md"],
        }
    )

    result = evaluate_retrieval(retriever, golden, k=3)

    assert [row["reciprocal_rank"] for row in result["per_query"]] == [1.0, 1 / 3, 0.0]
    assert result["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert result["precision_at_1"] == pytest.approx(1 / 3)


def test_report_has_the_metrics_and_per_question_rows():
    golden = [{"question": "Where can I pay?", "relevant_sources": ["general_info.md"]}]
    result = evaluate_retrieval(FakeRetriever({"Where can I pay?": ["general_info.md"]}), golden, k=1)

    report = render_report(result, chat_latency=None)

    assert "Mean Precision@1: 1.00" in report
    assert "Mean Recall@1: 1.00" in report
    assert "| Where can I pay? | single_fact | 1.00 | 1.00 |" in report


def test_only_questions_with_a_relevant_doc_are_scored_for_retrieval():
    golden = [
        {"question": "Where can I pay?", "relevant_sources": ["general_info.md"]},
        {"question": "Book me a taxi", "expected": "decline", "relevant_sources": []},
    ]

    assert [case["question"] for case in answerable(golden)] == ["Where can I pay?"]


def test_k_sweep_recommends_the_smallest_k_with_the_best_recall():
    golden = [
        {"question": "q1", "relevant_sources": ["a.md"]},
        {"question": "q2", "relevant_sources": ["a.md"]},
    ]
    # q2's answer is only ranked 3rd: K=2 misses it, K=4 adds nothing over K=3.
    retriever = FakeRetriever({"q1": ["a.md", "x.md", "y.md", "z.md"], "q2": ["x.md", "y.md", "a.md", "z.md"]})

    sweep = sweep_k(retriever, golden, [2, 3, 4])

    assert [result["mean_recall"] for result in sweep] == [0.5, 1.0, 1.0]
    assert recommend_k(sweep) == 3


def test_report_shows_the_k_sweep_and_flags_a_different_current_k():
    golden = [{"question": "q", "relevant_sources": ["a.md"]}]
    retriever = FakeRetriever({"q": ["x.md", "a.md", "y.md"]})
    sweep = sweep_k(retriever, golden, [1, 2, 3])

    report = render_report(evaluate_retrieval(retriever, golden, k=3), chat_latency=None, sweep=sweep)

    assert "## Choosing K" in report
    assert "| 2 (recommended) |" in report
    assert "| 3 (current) |" in report
    assert "RETRIEVER_K is 3 - set it to 2." in report
