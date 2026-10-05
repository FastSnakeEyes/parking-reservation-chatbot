import hashlib

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pymilvus import MilvusClient

from app.db.repositories.base import AbstractVectorRepository


class MilvusRepository(AbstractVectorRepository):
    def __init__(
        self,
        client: MilvusClient,
        embeddings: Embeddings,
        collection_name: str,
        embedding_model: str,
    ):
        self.client = client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def create_collection(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")

        if self.client.has_collection(self.collection_name):
            description = self.client.describe_collection(
                collection_name=self.collection_name,
            )

            field = next(
                (
                    field
                    for field in description["fields"]
                    if field["name"] == "vector"
                ),
                None,
            )

            if field is None:
                raise ValueError("Collection has no vector field")

            if int(field["params"]["dim"]) != dimension:
                raise ValueError(
                    "Collection and embedding dimensions differ"
                )

            return

        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dimension,
            id_type="string",
            max_length=64,
            auto_id=False,
            metric_type="COSINE",
            enable_dynamic_field=True,
        )

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 64,
    ) -> list[str]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        documents = [
            doc for doc in documents if doc.page_content.strip()
        ]
        ids = []

        for offset in range(0, len(documents), batch_size):
            batch = documents[offset:offset + batch_size]

            vectors = self.embeddings.embed_documents(
                [doc.page_content for doc in batch]
            )

            if len(vectors) != len(batch):
                raise ValueError("Embedding count differs from documents")

            if offset == 0:
                self.create_collection(len(vectors[0]))

            rows = []

            for document, vector in zip(batch, vectors):
                identity = document.metadata.get("id")

                if identity is None:
                    identity = (
                        f"{document.metadata.get('source', '')}:"
                        f"{document.page_content}"
                    )

                document_id = hashlib.sha256(
                    str(identity).encode("utf-8")
                ).hexdigest()

                rows.append({
                    "id": document_id,
                    "vector": vector,
                    "text": document.page_content,
                    "metadata": document.metadata,
                    "embedding_model": self.embedding_model,
                })

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

        vector = self.embeddings.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=k,
            output_fields=["text", "metadata", "embedding_model"],
            consistency_level="Strong",
        )

        documents = []

        for hit in results[0]:
            entity = hit["entity"]

            if entity.get("embedding_model") != self.embedding_model:
                raise ValueError(
                    "Embedding model mismatch. Use a separate collection."
                )

            documents.append(
                Document(
                    page_content=entity["text"],
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