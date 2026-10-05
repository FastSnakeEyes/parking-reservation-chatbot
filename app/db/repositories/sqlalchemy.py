import builtins
from typing import Any, Mapping

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Base
from app.db.repositories.base import AbstractSQLRepository


class SQLAlchemyRepository[ModelT: Base](
    AbstractSQLRepository[ModelT]
):
    def __init__(
        self,
        session: AsyncSession,
        model: type[ModelT],
    ):
        self.session = session
        self.model = model

    async def get_by_id(
        self,
        identity: Any,
    ) -> ModelT | None:
        return await self.session.get(self.model, identity)

    async def list(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> builtins.list[ModelT]:
        if offset < 0 or limit < 1:
            raise ValueError(
                "offset must be >= 0 and limit must be > 0"
            )

        statement = (
            sa.select(self.model)
            .order_by(*sa.inspect(self.model).primary_key)
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.scalars(statement)
        return builtins.list(result.all())

    async def add(self, entity: ModelT) -> ModelT:
        self.session.add(entity)
        await self.session.flush()
        return entity

    async def update(
        self,
        identity: Any,
        values: Mapping[str, Any],
    ) -> ModelT | None:
        mapper = sa.inspect(self.model)

        allowed_fields = {
            attribute.key
            for attribute in mapper.column_attrs
            if not any(
                column.primary_key
                for column in attribute.columns
            )
        }

        invalid_fields = set(values) - allowed_fields

        if invalid_fields:
            raise ValueError(
                f"Unknown or protected fields: "
                f"{sorted(invalid_fields)}"
            )

        entity = await self.get_by_id(identity)

        if entity is None:
            return None

        for field, value in values.items():
            setattr(entity, field, value)

        await self.session.flush()
        return entity

    async def delete(self, identity: Any) -> bool:
        entity = await self.get_by_id(identity)

        if entity is None:
            return False

        await self.session.delete(entity)
        await self.session.flush()
        return True