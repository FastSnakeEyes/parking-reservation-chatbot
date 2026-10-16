"""Milvus vector store for the static docs, searched by SEARCH_TYPE - hybrid by default:
semantic vectors (embeddings) plus Milvus's built-in BM25 full-text search
(exact keywords such as "LPG" or "height limit"), merged by a weighted ranker
(see app/constants/retrieval.py for the measured choice). The BM25 field is
computed by Milvus itself from the chunk text.
SEARCH_TYPE=semantic or keyword runs one side alone, to compare them (make eval).

A collection created before hybrid search (no BM25 field) still works:
search falls back to semantic and logs a warning until it is re-ingested
with `python -m scripts.ingest_static --recreate`.
"""

import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, WeightedRanker

from app.constants import (
    BM25_ANALYZER,
    HYBRID_CANDIDATES,
    KEYWORD_WEIGHT,
    SEMANTIC_WEIGHT,
    SPARSE_FIELD,
    TEXT_FIELD,
    TEXT_MAX_LENGTH,
    VECTOR_FIELD,
)
from app.db.repositories.base import AbstractVectorRepository
from app.enums import SearchType
from app.logger import get_logger

logger = get_logger(__name__)


class MilvusRepository(AbstractVectorRepository):
    def __init__(
        self,
        client: MilvusClient,
        embeddings: Embeddings,
        collection_name: str,
        embedding_model: str,
        search_type: SearchType = SearchType.HYBRID,
    ):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.search_type = search_type

    def create_collection(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")

        if self.client.has_collection(self.collection_name):
            description = self.client.describe_collection(
                collection_name=self.collection_name,
            )

            field = next(
                (field for field in description["fields"] if field["name"] == VECTOR_FIELD),
                None,
            )

            if field is None:
                raise ValueError("Collection has no vector field")

            if int(field["params"]["dim"]) != dimension:
                raise ValueError("Collection and embedding dimensions differ")

            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(
            TEXT_FIELD,
            DataType.VARCHAR,
            max_length=TEXT_MAX_LENGTH,
            enable_analyzer=True,
            analyzer_params=BM25_ANALYZER,
        )
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        # Milvus fills the BM25 field from `text` on every insert/upsert.
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=[TEXT_FIELD],
                output_field_names=[SPARSE_FIELD],
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(VECTOR_FIELD, index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(SPARSE_FIELD, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def drop_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

    def _is_hybrid(self) -> bool:
        """Whether the collection has the BM25 field (created with hybrid search)."""
        description = self.client.describe_collection(collection_name=self.collection_name)
        return any(field["name"] == SPARSE_FIELD for field in description["fields"])

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 64,
    ) -> list[str]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        documents = [doc for doc in documents if doc.page_content.strip()]
        ids = []

        for offset in range(0, len(documents), batch_size):
            batch = documents[offset : offset + batch_size]

            vectors = self.embeddings.embed_documents([doc.page_content for doc in batch])

            if len(vectors) != len(batch):
                raise ValueError("Embedding count differs from documents")

            if offset == 0:
                self.create_collection(len(vectors[0]))

            rows = []

            for document, vector in zip(batch, vectors, strict=True):
                identity = document.metadata.get("id")

                if identity is None:
                    identity = f"{document.metadata.get('source', '')}:{document.page_content}"

                document_id = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()

                rows.append(
                    {
                        "id": document_id,
                        VECTOR_FIELD: vector,
                        TEXT_FIELD: document.page_content,
                        "metadata": document.metadata,
                        "embedding_model": self.embedding_model,
                    }
                )

            self.client.upsert(
                collection_name=self.collection_name,
                data=rows,
            )

            ids.extend(row["id"] for row in rows)

        return ids

    def search(self, query: str, k: int = 4) -> list[Document]:
        if not query.strip():
            raise ValueError("query must not be empty")

        if k < 1:
            raise ValueError("k must be positive")

        if not self.client.has_collection(self.collection_name):
            raise RuntimeError("Collection does not exist. Run ingestion.")

        output_fields = [TEXT_FIELD, "metadata", "embedding_model"]
        search_type = self.search_type

        if search_type != SearchType.SEMANTIC and not self._is_hybrid():
            logger.warning(
                "Collection %s has no BM25 field - semantic search instead of %s. Re-ingest with "
                "`python -m scripts.ingest_static --recreate` to enable it.",
                self.collection_name,
                search_type,
            )
            search_type = SearchType.SEMANTIC

        if search_type == SearchType.KEYWORD:
            # BM25 alone: Milvus tokenizes the raw query itself, so no embedding call.
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field=SPARSE_FIELD,
                limit=k,
                output_fields=output_fields,
                consistency_level="Strong",
            )
        elif search_type == SearchType.HYBRID:
            vector = self.embeddings.embed_query(query)
            candidates = max(k, HYBRID_CANDIDATES)
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[
                    AnnSearchRequest(data=[vector], anns_field=VECTOR_FIELD, param={}, limit=candidates),
                    AnnSearchRequest(data=[query], anns_field=SPARSE_FIELD, param={}, limit=candidates),
                ],
                ranker=WeightedRanker(SEMANTIC_WEIGHT, KEYWORD_WEIGHT),
                limit=k,
                output_fields=output_fields,
                consistency_level="Strong",
            )
        else:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[self.embeddings.embed_query(query)],
                anns_field=VECTOR_FIELD,
                limit=k,
                output_fields=output_fields,
                consistency_level="Strong",
            )

        documents = []

        for hit in results[0]:
            entity = hit["entity"]

            if entity.get("embedding_model") != self.embedding_model:
                raise ValueError("Embedding model mismatch. Use a separate collection.")

            documents.append(
                Document(
                    page_content=entity[TEXT_FIELD],
                    metadata={
                        **entity.get("metadata", {}),
                        "vector_store_id": str(hit["id"]),
                    },
                )
            )

        return documents

    def delete_documents(self, ids: list[str]) -> None:
        if ids and self.client.has_collection(self.collection_name):
            self.client.delete(
                collection_name=self.collection_name,
                ids=ids,
            )

    def load_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(
                collection_name=self.collection_name,
            )

    def close(self) -> None:
        self.client.close()
