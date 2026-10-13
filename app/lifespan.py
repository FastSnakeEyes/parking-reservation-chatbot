import asyncio
import os
from contextlib import AsyncExitStack, asynccontextmanager, suppress

from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.concurrency import run_in_threadpool

from app.ai.chat_cleanup import run_chat_cleanup
from app.ai.graph import ParkingGraph
from app.ai.guardrails import get_injection_classifier, get_pii_guard
from app.config import check_production_secrets, settings
from app.constants import LLM_MAX_RETRIES, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS
from app.dependencies import create_vector_repository
from app.logger import get_logger

logger = get_logger(__name__)


def _load_injection_classifier() -> None:
    """With USE_INJECTION_MODEL on, load the classifier at startup: a missing
    backend (PyTorch isn't a default dependency) then stops the app with a
    clear error, instead of failing every chat message at runtime."""
    if not settings.use_injection_model:
        return

    try:
        get_injection_classifier()
    except Exception as exc:
        raise RuntimeError(
            "USE_INJECTION_MODEL=true, but the prompt-injection classifier "
            f"({settings.injection_model}) could not be loaded. It needs PyTorch "
            "(e.g. `uv add torch`); or set USE_INJECTION_MODEL=false."
        ) from exc

    logger.info("Prompt-injection classifier loaded (%s)", settings.injection_model)


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


async def _cancel(task: asyncio.Task) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up (env=%s)", settings.app_env.value)
    check_production_secrets(settings)

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

        llm_options = {
            "model": settings.llm_model,
            "api_key": api_key,
            "temperature": LLM_TEMPERATURE,
            "timeout": LLM_TIMEOUT_SECONDS,
            "max_retries": LLM_MAX_RETRIES,
        }
        llm = ChatOpenAI(
            **llm_options,
            # The user/admin agents' tools share one request-scoped AsyncSession;
            # LangGraph's ToolNode runs multiple tool calls from one turn
            # concurrently (asyncio.gather), which that session can't survive.
            model_kwargs={"parallel_tool_calls": False},
        )
        # History summarization calls the model without tools, and OpenAI rejects
        # parallel_tool_calls on a request without tools - so it gets its own client.
        summary_llm = ChatOpenAI(**llm_options)

        # Loads spaCy/Presidio once at startup instead of on the first request.
        await run_in_threadpool(get_pii_guard)
        await run_in_threadpool(_load_injection_classifier)

        checkpointer = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(settings.database_url_psycopg)
        )
        await checkpointer.setup()  # idempotent; creates checkpoint tables on first run
        # Built once; per-request dependencies go in through ChatContext.
        parking_graph = ParkingGraph(llm, checkpointer, redis, summary_llm=summary_llm)

        # Pushed after the engine/checkpointer, so the stack cancels it before closing them.
        cleanup_task = asyncio.create_task(run_chat_cleanup(engine, checkpointer, parking_graph.lock))
        stack.push_async_callback(_cancel, cleanup_task)

        app.state.db_engine = engine
        app.state.redis = redis
        app.state.vector_repository = vector_repository
        app.state.parking_graph = parking_graph

        logger.info("Startup complete")

        yield

        logger.info("Shutting down")
