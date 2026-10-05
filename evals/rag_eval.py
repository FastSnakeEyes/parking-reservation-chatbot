"""Stage 1 evaluation: retrieval accuracy (Recall@K, Precision@K) against a
small hand-labelled golden set, plus a retrieval-latency measurement.

Prerequisite: the static docs must already be ingested into Milvus
(`uv run python -m scripts.ingest_static`) and OPENAI_API_KEY/Milvus
settings must point at that same collection - this is a manual evaluation
tool, not part of the (fully offline) pytest suite.

Usage:
    uv run python -m evals.rag_eval [--k 4] [--api-url http://localhost:8000] [--out evals/report.md]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
from app.config import settings
from app.dependencies import create_vector_repository

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_qa.json"


def load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def evaluate_retrieval(repository, golden_set: list[dict], k: int) -> dict:
    per_query = []

    for case in golden_set:
        relevant = set(case["relevant_sources"])

        started = time.perf_counter()
        documents = repository.search(case["question"], k=k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        retrieved_sources = [document.metadata.get("source", "unknown") for document in documents]
        relevant_retrieved = {source for source in retrieved_sources if source in relevant}

        precision = len(relevant_retrieved) / len(retrieved_sources) if retrieved_sources else 0.0
        recall = len(relevant_retrieved) / len(relevant) if relevant else 0.0

        per_query.append(
            {
                "question": case["question"],
                "relevant_sources": sorted(relevant),
                "retrieved_sources": retrieved_sources,
                "precision": precision,
                "recall": recall,
                "latency_ms": elapsed_ms,
            }
        )

    return {
        "k": k,
        "per_query": per_query,
        "mean_precision": statistics.mean(row["precision"] for row in per_query),
        "mean_recall": statistics.mean(row["recall"] for row in per_query),
        "mean_latency_ms": statistics.mean(row["latency_ms"] for row in per_query),
        "p95_latency_ms": (
            statistics.quantiles([row["latency_ms"] for row in per_query], n=20)[18]
            if len(per_query) >= 2
            else per_query[0]["latency_ms"]
        ),
    }


def measure_chat_latency(api_url: str, golden_set: list[dict], rounds: int = 1) -> dict | None:
    """End-to-end /chat latency, if a running API URL was given."""
    try:
        with httpx.Client(base_url=api_url, timeout=30) as client:
            client.get("/health").raise_for_status()

            samples = []

            for case in golden_set[:5] * rounds:
                started = time.perf_counter()
                response = client.post("/chat", json={"question": case["question"]})
                response.raise_for_status()
                samples.append((time.perf_counter() - started) * 1000)
    except httpx.HTTPError as exc:
        print(f"Skipping /chat latency check ({exc})")
        return None

    return {
        "samples": len(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": statistics.quantiles(samples, n=20)[18] if len(samples) >= 2 else samples[0],
    }


def render_report(retrieval: dict, chat_latency: dict | None) -> str:
    lines = [
        "# RAG evaluation report",
        "",
        f"Golden set size: {len(retrieval['per_query'])} questions, K={retrieval['k']}",
        "",
        "## Retrieval accuracy",
        "",
        f"- Mean Precision@{retrieval['k']}: {retrieval['mean_precision']:.2f}",
        f"- Mean Recall@{retrieval['k']}: {retrieval['mean_recall']:.2f}",
        "",
        "## Retrieval latency",
        "",
        f"- Mean: {retrieval['mean_latency_ms']:.1f} ms",
        f"- p95: {retrieval['p95_latency_ms']:.1f} ms",
        "",
    ]

    if chat_latency is not None:
        lines += [
            "## End-to-end /chat latency",
            "",
            f"- Samples: {chat_latency['samples']}",
            f"- Mean: {chat_latency['mean_ms']:.1f} ms",
            f"- p95: {chat_latency['p95_ms']:.1f} ms",
            "",
        ]

    lines += [
        "## Per-question detail",
        "",
        "| Question | Precision | Recall | Latency (ms) | Retrieved sources |",
        "| --- | --- | --- | --- | --- |",
    ]

    for row in retrieval["per_query"]:
        sources = ", ".join(row["retrieved_sources"]) or "-"
        lines.append(
            f"| {row['question']} | {row['precision']:.2f} | {row['recall']:.2f} | "
            f"{row['latency_ms']:.1f} | {sources} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=settings.retriever_k)
    parser.add_argument("--api-url", default=None, help="If set, also measures end-to-end /chat latency")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "report.md")
    args = parser.parse_args()

    golden_set = load_golden_set()
    repository = create_vector_repository()

    try:
        repository.load_collection()
        retrieval = evaluate_retrieval(repository, golden_set, args.k)
    finally:
        repository.close()

    chat_latency = measure_chat_latency(args.api_url, golden_set) if args.api_url else None

    report = render_report(retrieval, chat_latency)
    args.out.write_text(report, encoding="utf-8")

    print(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
