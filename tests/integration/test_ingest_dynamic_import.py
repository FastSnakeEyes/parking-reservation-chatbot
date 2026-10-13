"""scripts/ingest_dynamic.py's main() against the test Postgres: it loads all
the dynamic data, and re-running it updates rows instead of duplicating them."""

import pytest
import sqlalchemy as sa
from app.config import settings
from app.db.models import CapacityOverride, Parking, ParkingSpaceType, Price, SpecialHours, WorkingHours
from pydantic import SecretStr
from scripts import ingest_dynamic

MODELS = (Parking, ParkingSpaceType, WorkingHours, SpecialHours, Price, CapacityOverride)
SLUGS = [parking["slug"] for parking in ingest_dynamic.build_parkings()]


async def _counts(engine) -> dict[str, int]:
    """Rows belonging to the imported parkings (other tests' parkings excluded)."""
    async with engine.connect() as conn:
        ids = (await conn.execute(sa.select(Parking.id).where(Parking.slug.in_(SLUGS)))).scalars().all()
        return {
            model.__tablename__: await conn.scalar(
                sa.select(sa.func.count())
                .select_from(model)
                .where((model.id if model is Parking else model.parking_id).in_(ids))
            )
            for model in MODELS
        }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_import_loads_everything_and_is_idempotent(engine, postgres_container, monkeypatch, capsys):
    monkeypatch.setattr(settings, "database_url_override", SecretStr(postgres_container))

    await ingest_dynamic.main()
    first = await _counts(engine)
    await ingest_dynamic.main()
    second = await _counts(engine)

    assert first == {
        "parkings": 5,
        "parking_space_types": 10,
        "working_hours": 35,
        "special_hours": 12,
        "prices": 22,
        "capacity_overrides": 2,
    }
    assert second == first  # a re-run updates in place
    assert "Transaction committed." in capsys.readouterr().out
