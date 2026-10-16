"""Stage 1 evaluation: retrieval accuracy (Recall@K, Precision@K, MRR) against a
small hand-labelled golden set, plus a retrieval-latency measurement, and the
same metrics for K=2..5 to choose RETRIEVER_K.

Prerequisite: the static docs must already be ingested into Milvus
(`uv run python -m scripts.ingest_static`) and OPENAI_API_KEY/Milvus
settings must point at that same collection - this is a manual evaluation
tool, not part of the (fully offline) pytest suite.

Usage:
    uv run python -m evals.rag_eval [--k 3] [--k-range 2 5] [--api-url http://localhost:8000] [--out evals/report.md]
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
CHAT_REQUEST_SPACING_SECONDS = 5.5
K_SWEEP_RANGE = (2, 5)  # the K values compared in "Choosing K"


def connect_vector_repository():
    """The Milvus repository, or a one-line explanation if it can't be reached -
    usually because .env.development's in-Docker hostname is used on the host."""
    from pymilvus.exceptions import MilvusException

    try:
        return create_vector_repository()
    except MilvusException as exc:
        raise SystemExit(
            f"Can't reach Milvus at {settings.milvus_uri}. Running on the host? Use `make eval` / "
            "`make eval-evidently`, or set VECTOR_DB_URI=http://localhost:19530 (the published port) - "
            "and make sure the stack is up."
        ) from exc


def load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def answerable(golden_set: list[dict]) -> list[dict]:
    """The questions with a relevant doc. The others (out of scope, private data,
    injection, not in the docs) should be declined: they have no right chunk to
    retrieve, so they are scored by evals/evidently_eval.py only."""
    return [case for case in golden_set if case["relevant_sources"]]


def evaluate_retrieval(repository, golden_set: list[dict], k: int) -> dict:
    per_query = []

    for case in golden_set:
        relevant = set(case["relevant_sources"])

        started = time.perf_counter()
        documents = repository.search(case["question"], k=k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        retrieved_sources = [document.metadata.get("source", "unknown") for document in documents]

        # Precision@K: share of the K retrieved chunks that come from a relevant doc.
        relevant_chunks = sum(source in relevant for source in retrieved_sources)
        precision = relevant_chunks / len(retrieved_sources) if retrieved_sources else 0.0
        # Rank of the first chunk from a relevant doc (None if none in the top K):
        # MRR rewards putting the answer first, which Precision@K can't show.
        first_rank = next((rank for rank, source in enumerate(retrieved_sources, 1) if source in relevant), None)
        # Recall@K: share of the relevant docs with at least one chunk in the top K.
        recall = len(relevant & set(retrieved_sources)) / len(relevant) if relevant else 0.0

        per_query.append(
            {
                "question": case["question"],
                "category": case.get("category", "single_fact"),
                "relevant_sources": sorted(relevant),
                "retrieved_sources": retrieved_sources,
                "precision": precision,
                "recall": recall,
                "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
                "hit_at_1": first_rank == 1,
                "latency_ms": elapsed_ms,
            }
        )

    return {
        "k": k,
        "per_query": per_query,
        "mean_precision": statistics.mean(row["precision"] for row in per_query),
        "mean_recall": statistics.mean(row["recall"] for row in per_query),
        "mrr": statistics.mean(row["reciprocal_rank"] for row in per_query),
        "precision_at_1": statistics.mean(row["hit_at_1"] for row in per_query),
        "mean_latency_ms": statistics.mean(row["latency_ms"] for row in per_query),
        "p95_latency_ms": (
            statistics.quantiles([row["latency_ms"] for row in per_query], n=20)[18]
            if len(per_query) >= 2
            else per_query[0]["latency_ms"]
        ),
    }


def sweep_k(repository, golden_set: list[dict], ks: list[int]) -> list[dict]:
    """The same retrieval metrics at each K, to choose RETRIEVER_K."""
    return [evaluate_retrieval(repository, golden_set, k) for k in ks]


def recommend_k(sweep: list[dict]) -> int:
    """The smallest K that reaches the best recall: a larger K only adds chunks
    the model has to read past, never an answer it was missing."""
    best_recall = max(result["mean_recall"] for result in sweep)
    return min(result["k"] for result in sweep if result["mean_recall"] == best_recall)


def render_k_sweep(sweep: list[dict], current_k: int) -> list[str]:
    recommended = recommend_k(sweep)
    lines = [
        "## Choosing K",
        "",
        "| K | Precision@K | Recall@K | MRR@K | Precision@1 | Mean latency (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in sweep:
        notes = [
            note
            for note, applies in (("current", result["k"] == current_k), ("recommended", result["k"] == recommended))
            if applies
        ]
        label = f"{result['k']} ({', '.join(notes)})" if notes else str(result["k"])
        lines.append(
            f"| {label} | {result['mean_precision']:.2f} | {result['mean_recall']:.2f} | {result['mrr']:.2f} | "
            f"{result['precision_at_1']:.2f} | {result['mean_latency_ms']:.1f} |"
        )
    lines += [
        "",
        f"Recommended K={recommended}: the smallest K with the best recall. A larger K only adds chunks the "
        "model has to read past (Precision@K falls), and a smaller one misses answers. "
        + (
            "RETRIEVER_K already uses it."
            if recommended == current_k
            else f"RETRIEVER_K is {current_k} - set it to {recommended}."
        ),
        "",
    ]
    return lines


def measure_chat_latency(api_url: str, golden_set: list[dict], rounds: int = 1) -> dict | None:
    """End-to-end /chat latency, if a running API URL was given."""
    try:
        with httpx.Client(base_url=api_url, timeout=60) as client:
            client.get("/health").raise_for_status()
            client.get("/").raise_for_status()  # the session cookie /chat requires

            samples = []

            for index, case in enumerate(golden_set[:5] * rounds):
                if index:
                    # Stay under the per-session limit (2 requests / 10 s).
                    time.sleep(CHAT_REQUEST_SPACING_SECONDS)
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


def render_report(
    retrieval: dict, chat_latency: dict | None, sweep: list[dict] | None = None, golden_size: int | None = None
) -> str:
    answerable_count = len(retrieval["per_query"])
    size = (
        f"{golden_size} questions; retrieval is scored on the {answerable_count} with a relevant doc"
        if golden_size and golden_size != answerable_count
        else f"{answerable_count} questions"
    )
    lines = [
        "# RAG evaluation report",
        "",
        f"Golden set: {size}, K={retrieval['k']}",
        "",
        "## Retrieval accuracy",
        "",
        f"- Mean Precision@{retrieval['k']}: {retrieval['mean_precision']:.2f}",
        f"- Mean Recall@{retrieval['k']}: {retrieval['mean_recall']:.2f}",
        f"- MRR@{retrieval['k']}: {retrieval['mrr']:.2f} (1.00 = the relevant doc is always ranked first)",
        f"- Precision@1: {retrieval['precision_at_1']:.2f} (share of questions whose top chunk is relevant)",
        "",
        *(render_k_sweep(sweep, retrieval["k"]) if sweep else []),
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
        "## Method",
        "",
        "- Relevance is labelled per source document (`golden_qa.json`); the static docs are",
        "  split into chunks, so one document can fill several of the K results.",
        f"- **Precision@{retrieval['k']}**: share of the K retrieved chunks that come from a relevant document.",
        f"- **Recall@{retrieval['k']}**: share of the relevant documents with at least one chunk in the top K.",
        "- **MRR** (mean reciprocal rank): 1 / the rank of the first relevant chunk, averaged; it shows"
        " whether the answer comes first. **Precision@1** is the share of questions where it does.",
        "- Precision@K is capped below 1: most answers sit in one chunk, but K chunks are always"
        " returned, and small docs (rules.md has 2 chunks) can't fill K slots. Recall and MRR are the"
        " better signals for answer quality.",
        "- Retrieval latency includes embedding the question through the OpenAI embeddings API,",
        "  which dominates it; the vector search itself is a few milliseconds.",
        "- End-to-end /chat latency is a real chat turn (agent, tools, model calls), spaced to stay",
        "  under the per-session rate limit.",
        "",
        "## Per-question detail",
        "",
        "| Question | Category | Precision | Recall | Latency (ms) | Retrieved sources |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in retrieval["per_query"]:
        sources = ", ".join(row["retrieved_sources"]) or "-"
        lines.append(
            f"| {row['question']} | {row['category']} | {row['precision']:.2f} | {row['recall']:.2f} | "
            f"{row['latency_ms']:.1f} | {sources} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=settings.retriever_k)
    parser.add_argument(
        "--k-range",
        type=int,
        nargs=2,
        default=list(K_SWEEP_RANGE),
        metavar=("MIN", "MAX"),
        help="also compare every K in this range (inclusive) and recommend one",
    )
    parser.add_argument("--api-url", default=None, help="If set, also measures end-to-end /chat latency")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "report.md")
    args = parser.parse_args()

    golden_set = load_golden_set()
    scored_set = answerable(golden_set)
    repository = connect_vector_repository()
    low, high = args.k_range

    try:
        repository.load_collection()
        retrieval = evaluate_retrieval(repository, scored_set, args.k)
        sweep = sweep_k(repository, scored_set, list(range(low, high + 1)))
    finally:
        repository.close()

    chat_latency = measure_chat_latency(args.api_url, scored_set) if args.api_url else None

    report = render_report(retrieval, chat_latency, sweep, golden_size=len(golden_set))
    # Keep the Evidently section (evals/evidently_eval.py) if a previous run appended one.
    existing = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
    marker = "\n## Evidently AI:"
    if marker in existing:
        report = report.rstrip() + "\n" + marker + existing.split(marker, 1)[1]
    args.out.write_text(report, encoding="utf-8")

    print(report)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
