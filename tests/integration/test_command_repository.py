"""CommandRepository: the durable outbox GraphRunner (Phase 2) will enqueue
into and the in-process worker will claim from - idempotent enqueue and
exactly-once claiming under concurrency."""

import asyncio
from uuid import uuid4

import pytest
from app.db.repositories.command import CommandRepository
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enqueue_is_idempotent(real_session):
    repository = CommandRepository(real_session)
    key = f"enqueue-{uuid4()}"

    first = await repository.enqueue(key, "resume_user", "thread-1", {"answer": "yes"})
    await real_session.commit()
    second = await repository.enqueue(key, "resume_user", "thread-1", {"answer": "yes"})
    await real_session.commit()

    assert first.id == second.id


async def _claim(engine: AsyncEngine):
    async with AsyncSession(engine, expire_on_commit=False) as session:
        repository = CommandRepository(session)
        return await repository.claim_next()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_claimers_never_double_claim(engine: AsyncEngine, real_session):
    repository = CommandRepository(real_session)
    key_a = f"claim-{uuid4()}"
    key_b = f"claim-{uuid4()}"
    await repository.enqueue(key_a, "resume_user", "thread-a", {})
    await repository.enqueue(key_b, "resume_user", "thread-b", {})
    await real_session.commit()

    first, second = await asyncio.gather(_claim(engine), _claim(engine))

    assert first is not None
    assert second is not None
    assert first.id != second.id
    assert {first.status, second.status} == {"claimed"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claim_next_returns_none_when_empty(real_session):
    """real_session's Postgres container is shared (committed, not rolled
    back) across every test in this module, so other tests' pending
    commands may still be sitting in the table - drain them first rather
    than assuming the table starts empty."""
    repository = CommandRepository(real_session)

    while await repository.claim_next() is not None:
        pass

    command = await repository.claim_next()

    assert command is None
