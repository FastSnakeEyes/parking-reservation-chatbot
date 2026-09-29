from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    app.state.redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )
    yield
    await app.state.redis.aclose()
    await app.state.db_engine.dispose()


app = FastAPI(title="FastAPI + Postgres + Redis", lifespan=lifespan)


async def get_db():
    async with AsyncSession(
        app.state.db_engine,
        expire_on_commit=False,
    ) as session:
        yield session


async def get_redis():
    return app.state.redis


@app.get("/")
async def root():
    return {"environment": settings.app_env.value, "status": "ok"}


@app.get("/health")
async def health(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    try:
        await db.execute(text("SELECT 1"))
        await redis.ping()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="A backing service is unavailable",
        ) from exc

    return {"api": "ok", "postgres": "ok", "redis": "ok"}