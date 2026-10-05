import builtins
from abc import ABC, abstractmethod
from typing import Any, Mapping

from langchain_core.documents import Document


class AbstractVectorRepository(ABC):
    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        ...

    @abstractmethod
    def add_documents(
        self,
        documents: builtins.list[Document],
        batch_size: int = 64,
    ) -> builtins.list[str]:
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 4,
    ) -> builtins.list[Document]:
        ...

    @abstractmethod
    def delete_documents(self, ids: builtins.list[str]) -> None:
        ...

    @abstractmethod
    def load_collection(self) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class AbstractSQLRepository[EntityT](ABC):
    @abstractmethod
    async def get_by_id(
        self,
        identity: Any,
    ) -> EntityT | None:
        ...

    @abstractmethod
    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> builtins.list[EntityT]:
        ...

    @abstractmethod
    async def add(self, entity: EntityT) -> EntityT:
        ...

    @abstractmethod
    async def update(
        self,
        identity: Any,
        values: Mapping[str, Any],
    ) -> EntityT | None:
        ...

    @abstractmethod
    async def delete(self, identity: Any) -> bool:
        ...