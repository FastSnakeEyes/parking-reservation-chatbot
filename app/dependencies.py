import time
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.parking_chat import ParkingChat
from app.config import settings
from app.db.repositories.base import AbstractVectorRepository
from app.db.repositories.milvus import MilvusRepository


def create_vector_repository() -> MilvusRepository:
    api_key = settings.openai_api_key.get_secret_value()

    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
    )

    client = MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
    )

    return MilvusRepository(
        client=client,
        embeddings=embeddings,
        collection_name=settings.milvus_collection,
        embedding_model=settings.embedding_model,
    )


def get_vector_repository(
    request: Request,
) -> AbstractVectorRepository:
    return request.app.state.vector_repository


async def get_sql_session(
    request: Request,
) -> AsyncIterator[AsyncSession]:
    async with AsyncSession(
        request.app.state.db_engine,
        expire_on_commit=False,
    ) as session:
        yield session


SQLSessionDependency = Annotated[
    AsyncSession,
    Depends(get_sql_session),
]


def get_redis(request: Request) -> Redis:
    return request.app.state.redis


RedisDependency = Annotated[
    Redis,
    Depends(get_redis),
]


async def require_cookie(request: Request) -> str:
    cookie_id = request.cookies.get(settings.cookie_name)

    if not cookie_id:
        raise HTTPException(
            status_code=400,
            detail=f"Missing cookie: {settings.cookie_name}",
        )

    try:
        return str(UUID(cookie_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid chat cookie. Reload the homepage.",
        ) from exc


CookieDependency = Annotated[
    str,
    Depends(require_cookie),
]


async def cookie_rate_limit(
    request: Request,
    user_id: CookieDependency,
) -> None:
    """At most rate_limit_times requests per rate_limit_seconds per user."""
    redis_conn = request.app.state.redis
    key = f"rate_limit:{user_id}"

    count = await redis_conn.incr(key)
    if count == 1:
        await redis_conn.expire(key, settings.rate_limit_seconds)

    if count > settings.rate_limit_times:
        ttl = await redis_conn.ttl(key)
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment.",
            headers={"Retry-After": str(max(ttl, 1))},
        )


async def global_user_limit(
    request: Request,
    user_id: CookieDependency,
) -> None:
    """At most max_active_users users active within active_user_ttl seconds."""
    redis_conn = request.app.state.redis

    key = "active_users"
    now = int(time.time())

    # 1. Remove users whose last activity is older than active_user_ttl
    entries = await redis_conn.hgetall(key)
    stale_users = [uid for uid, last_seen in entries.items() if now - int(last_seen) > settings.active_user_ttl]
    if stale_users:
        await redis_conn.hdel(key, *stale_users)

    # 2. Check BEFORE adding, so rejected users don't take a slot
    is_active = await redis_conn.hexists(key, user_id)
    active_count = await redis_conn.hlen(key)

    if not is_active and active_count >= settings.max_active_users:
        raise HTTPException(
            status_code=429,
            detail=f"Too many active users (max {settings.max_active_users})",
            headers={"Retry-After": str(settings.active_user_ttl)},
        )

    # 3. Register / refresh this user's activity
    await redis_conn.hset(key, user_id, str(now))
    await redis_conn.expire(key, settings.active_user_ttl)


def get_chat_service(
    request: Request,
    session: SQLSessionDependency,
) -> ParkingChat:
    return ParkingChat(
        session=session,
        redis=request.app.state.redis,
        vector_repository=request.app.state.vector_repository,
        agent=request.app.state.chat_agent,
        approval_pipeline=request.app.state.approval_pipeline,
    )


ChatServiceDependency = Annotated[
    ParkingChat,
    Depends(get_chat_service),
]

VectorRepositoryDependency = Annotated[
    AbstractVectorRepository,
    Depends(get_vector_repository),
]
