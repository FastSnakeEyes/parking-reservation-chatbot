from enum import StrEnum


class SearchType(StrEnum):
    """How app/db/repositories/milvus.py searches the static docs (SEARCH_TYPE)."""

    HYBRID = "hybrid"  # semantic vectors + BM25 keywords, merged by a weighted ranker
    SEMANTIC = "semantic"  # embedding vectors only: matches meaning
    KEYWORD = "keyword"  # BM25 only: exact words, no embedding call
