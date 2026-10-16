import secrets
import time
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.graph import ParkingGraph
from app.config import settings
from app.constants import ADMIN_AUTH_MAX_FAILURES, ADMIN_AUTH_WINDOW_SECONDS
from app.db.repositories.milvus import MilvusRepository
from app.db.repositories.parking import ParkingRepository
from app.db.repositories.reservation import ReservationRepository
from app.logger import get_logger
from app.services.parking import ParkingService

logger = get_logger(__name__)


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
        search_type=settings.search_type,
    )


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
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing cookie: {settings.cookie_name}",
        )

    try:
        return str(UUID(cookie_id))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid chat cookie. Reload the homepage.",
        ) from exc


CookieDependency = Annotated[
    str,
    Depends(require_cookie),
]


async def require_admin(request: Request, x_admin_key: str = Header(default="")) -> None:
    """The admin API's guard (app/api/routers/admin.py): the X-Admin-Key header
    must match settings.admin_api_key. Compared in constant time, and a client
    IP with too many failed attempts is locked out for a while, so the key
    can't be guessed by timing or brute force."""
    redis_conn = request.app.state.redis
    client = request.client.host if request.client else "unknown"
    failures_key = f"admin_auth_failures:{client}"

    failures = int(await redis_conn.get(failures_key) or 0)
    if failures >= ADMIN_AUTH_MAX_FAILURES:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed admin key attempts. Try again later.",
            headers={"Retry-After": str(ADMIN_AUTH_WINDOW_SECONDS)},
        )

    expected = settings.admin_api_key.get_secret_value()
    if not secrets.compare_digest(x_admin_key.encode(), expected.encode()):
        if await redis_conn.incr(failures_key) == 1:
            await redis_conn.expire(failures_key, ADMIN_AUTH_WINDOW_SECONDS)
        logger.warning("Rejected an admin request with an invalid X-Admin-Key (client=%s)", client)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin key")


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
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
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
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many active users (max {settings.max_active_users})",
            headers={"Retry-After": str(settings.active_user_ttl)},
        )

    # 3. Register / refresh this user's activity
    await redis_conn.hset(key, user_id, str(now))
    await redis_conn.expire(key, settings.active_user_ttl)


def get_parking_service(
    request: Request,
    session: SQLSessionDependency,
) -> ParkingService:
    return ParkingService(
        parking_repository=ParkingRepository(session),
        reservation_repository=ReservationRepository(session),
        vector_repository=request.app.state.vector_repository,
    )


ParkingServiceDependency = Annotated[
    ParkingService,
    Depends(get_parking_service),
]


def get_parking_graph(request: Request) -> ParkingGraph:
    return request.app.state.parking_graph


ParkingGraphDependency = Annotated[
    ParkingGraph,
    Depends(get_parking_graph),
]
