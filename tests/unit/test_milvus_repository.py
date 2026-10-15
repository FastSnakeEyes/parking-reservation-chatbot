"""MilvusRepository (app/db/repositories/milvus.py) against an embedded
Milvus Lite database file, with deterministic fake embeddings - no Milvus
server or OpenAI calls."""

import pytest
from app.db.repositories.milvus import MilvusRepository
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from pymilvus import MilvusClient

DIMENSION = 16


@pytest.fixture
def repository(tmp_path):
    client = MilvusClient(str(tmp_path / "milvus.db"))
    repository = MilvusRepository(
        client=client,
        embeddings=DeterministicFakeEmbedding(size=DIMENSION),
        collection_name="test_static",
        embedding_model="fake",
    )
    repository.create_collection(DIMENSION)
    yield repository
    repository.close()


def _document(text: str, source: str, index: int) -> Document:
    return Document(page_content=text, metadata={"source": source, "chunk_index": index, "id": f"{source}:{index}"})


def test_added_documents_are_searchable_with_their_metadata(repository):
    texts = ["Opening hours are 07:00-22:00.", "EV charging costs extra.", "Pay at the exit machine."]
    repository.add_documents([_document(text, "rules.md", i) for i, text in enumerate(texts)])
    repository.load_collection()

    # Deterministic fake embeddings: an identical text is its own nearest neighbour.
    results = repository.search("EV charging costs extra.", k=2)

    assert results[0].page_content == "EV charging costs extra."
    assert results[0].metadata["source"] == "rules.md"
    assert len(results) == 2


def test_create_collection_rejects_a_different_dimension(repository):
    with pytest.raises(ValueError, match="dimensions differ"):
        repository.create_collection(DIMENSION * 2)


def test_create_collection_rejects_a_non_positive_dimension(repository):
    with pytest.raises(ValueError, match="positive"):
        repository.create_collection(0)


def test_hybrid_search_finds_an_exact_keyword(repository):
    """BM25 ranks the chunk with the keyword first. The fake embeddings carry no
    meaning, so the dense side alone couldn't - only the keyword side can."""
    texts = [
        "Opening hours are 07:00-22:00 on weekdays.",
        "Vehicles running on LPG are not allowed in the underground car park.",
        "Pay at the exit machine by card or cash.",
        "EV charging is billed separately from parking.",
    ]
    repository.add_documents([_document(text, "rules.md", i) for i, text in enumerate(texts)])
    repository.load_collection()

    results = repository.search("Is LPG allowed?", k=2)

    assert results[0].page_content == texts[1]


def test_a_collection_without_bm25_falls_back_to_dense_search(tmp_path):
    """A collection created before hybrid search keeps working until re-ingested."""
    client = MilvusClient(str(tmp_path / "old.db"))
    embeddings = DeterministicFakeEmbedding(size=DIMENSION)
    client.create_collection(
        collection_name="old_static",
        dimension=DIMENSION,
        id_type="string",
        max_length=64,
        auto_id=False,
        metric_type="COSINE",
        enable_dynamic_field=True,
    )
    client.insert(
        "old_static",
        [
            {
                "id": "a",
                "vector": embeddings.embed_query("EV charging costs extra."),
                "text": "EV charging costs extra.",
                "metadata": {"source": "rules.md"},
                "embedding_model": "fake",
            }
        ],
    )
    repository = MilvusRepository(client, embeddings, collection_name="old_static", embedding_model="fake")
    repository.load_collection()

    results = repository.search("EV charging costs extra.", k=1)

    assert results[0].page_content == "EV charging costs extra."
    repository.close()


def test_drop_collection_removes_it(repository):
    repository.drop_collection()

    assert not repository.client.has_collection(repository.collection_name)
