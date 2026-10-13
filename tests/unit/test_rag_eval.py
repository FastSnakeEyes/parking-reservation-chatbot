"""The RAG evaluation's metrics (evals/rag_eval.py), on a fake retriever."""

from evals.rag_eval import evaluate_retrieval, render_report
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


def test_report_has_the_metrics_and_per_question_rows():
    golden = [{"question": "Where can I pay?", "relevant_sources": ["general_info.md"]}]
    result = evaluate_retrieval(FakeRetriever({"Where can I pay?": ["general_info.md"]}), golden, k=1)

    report = render_report(result, chat_latency=None)

    assert "Mean Precision@1: 1.00" in report
    assert "Mean Recall@1: 1.00" in report
    assert "| Where can I pay? | 1.00 | 1.00 |" in report
