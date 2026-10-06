import os
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.concurrency import run_in_threadpool

from app.ai.graph import ApprovalPipeline
from app.ai.guardrails import get_pii_guard
from app.config import settings
from app.dependencies import create_vector_repository
from app.logger import get_logger

logger = get_logger(__name__)


def _configure_langsmith() -> None:
    """LangChain/LangGraph read these env vars directly to decide whether
    to trace; there's no SDK call to enable it. Must run before the first
    traced call (ChatOpenAI, graph.ainvoke, ...)."""
    if not settings.langsmith_tracing:
        logger.info("LangSmith tracing disabled")
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    if settings.langsmith_api_key is not None:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()

    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up (env=%s)", settings.app_env.value)

    if settings.redis_url is None:
        raise RuntimeError("REDIS_URL is required")

    api_key = settings.openai_api_key.get_secret_value()

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    _configure_langsmith()

    async with AsyncExitStack() as stack:
        engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
        )
        stack.push_async_callback(engine.dispose)

        redis = Redis.from_url(
            settings.redis_url.get_secret_value(),
            decode_responses=True,
        )
        stack.push_async_callback(redis.aclose)

        vector_repository = await run_in_threadpool(
            create_vector_repository,
        )
        stack.push_async_callback(
            run_in_threadpool,
            vector_repository.close,
        )

        # Static ingestion must have created the collection.
        # Load it once when the application starts.
        await run_in_threadpool(
            vector_repository.load_collection,
        )

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=api_key,
            temperature=0,
            timeout=30,
            max_retries=1,
        )

        # Loads spaCy/Presidio once at startup instead of on the first request.
        await run_in_threadpool(get_pii_guard)

        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.database_url_psycopg)
        )
        await checkpointer.setup()  # idempotent; creates checkpoint tables on first run
        approval_pipeline = ApprovalPipeline(engine, checkpointer)

        app.state.db_engine = engine
        app.state.redis = redis
        app.state.vector_repository = vector_repository
        app.state.llm = llm
        app.state.approval_pipeline = approval_pipeline

        logger.info("Startup complete")

        yield

        logger.info("Shutting down")
