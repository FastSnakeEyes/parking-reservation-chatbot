"""Vector search over the static docs (app/ai/tools.py's search_public_rules)."""

# The operator's brand name, stripped from search queries: every doc is about
# VilniusPark, and each one's intro chunk is full of the name - so a query like
# "cancellation policy VilniusPark" matched those generic intros instead of the
# chunk with the cancellation rules (found by the Evidently evaluation).
BRAND_NAME_PATTERN = r"\bvilnius\s*park\b"

# Milvus collection fields (app/db/repositories/milvus.py): the chunk text, its
# embedding, and its BM25 sparse vector, which Milvus computes from the text.
TEXT_FIELD = "text"
VECTOR_FIELD = "vector"
SPARSE_FIELD = "sparse"

# Hybrid search (app/db/repositories/milvus.py): semantic vector search and BM25
# keyword search each return this many candidates, merged by a weighted ranker.
# Chosen by measurement on the golden set (evals/report.md): this mix was the
# only one to find the relevant doc for every question at k=3 and k=4, without
# losing precision; equal weights or Reciprocal Rank Fusion each missed one.
HYBRID_CANDIDATES = 10
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
# BM25 tokenizer: lowercase, English stemming ("tickets" -> "ticket") and
# English stop words ("if", "the", "is" ...), so a question's filler words don't
# swamp its keywords - the default ("standard") kept them and made hybrid search
# worse than semantic alone. This is Milvus's built-in {"type": "english"} analyzer
# spelled out, because Milvus Lite (used by the tests) only accepts this form.
BM25_ANALYZER = {
    "tokenizer": "standard",
    "filter": ["lowercase", {"type": "stemmer", "language": "english"}, {"type": "stop", "stop_words": ["_english_"]}],
}
# Longest chunk text the collection stores (ingestion chunks are ~1000 characters).
TEXT_MAX_LENGTH = 8192
