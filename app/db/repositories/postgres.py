import builtins
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from app.db.models import Base
from app.db.repositories.sqlalchemy import SQLAlchemyRepository


class PostgresRepository[ModelT: Base](SQLAlchemyRepository[ModelT]):
    async def upsert_many(
        self,
        rows: builtins.list[dict[str, Any]],
    ) -> int:
        """Upsert rows using PostgreSQL primary-key conflicts."""
        table = self.model.__table__

        column_names = set(table.columns.keys())
        primary_keys = [column.name for column in table.primary_key]

        # Validate every row before issuing writes.
        for row in rows:
            unknown_fields = set(row) - column_names
            missing_keys = set(primary_keys) - set(row)

            if unknown_fields:
                raise ValueError(f"Unknown fields: {sorted(unknown_fields)}")

            if missing_keys:
                raise ValueError(f"Missing primary keys: {sorted(missing_keys)}")

        for row in rows:
            values = dict(row)

            # Preserve PostgreSQL time values such as "24:00".
            for name, value in values.items():
                column = table.c[name]

                if isinstance(column.type, sa.Time) and isinstance(value, str):
                    values[name] = sa.cast(
                        sa.literal(value, type_=sa.String()),
                        column.type,
                    )

            statement = insert(table).values(**values)

            updates = {name: statement.excluded[name] for name in row if name not in primary_keys}

            if updates:
                statement = statement.on_conflict_do_update(
                    index_elements=primary_keys,
                    set_=updates,
                )
            else:
                statement = statement.on_conflict_do_nothing(
                    index_elements=primary_keys,
                )

            await self.session.execute(statement)

        return len(rows)
