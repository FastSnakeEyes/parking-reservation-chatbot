from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.concurrency import run_in_threadpool

from app.ai.graph import ApprovalPipeline
from app.ai.guardrails import get_pii_guard
from app.api.routers.admin import router as admin_router
from app.api.routers.chat import router as chat_router
from app.api.routers.pages import router as pages_router
from app.config import settings
from app.dependencies import create_vector_repository

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.redis_url is None:
        raise RuntimeError("REDIS_URL is required")

    api_key = settings.openai_api_key.get_secret_value()

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

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

        yield


app = FastAPI(
    title="Parking chatbot",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(APP_DIR / "static")),
    name="static",
)

# Defines POST /chat using ParkingChat and SQL/vector tools.
app.include_router(chat_router)

# Defines the admin console API (Stage 2); the /admin page and
# ui/admin_app.py (Streamlit) are both clients of it.
app.include_router(admin_router)

# Defines GET /, GET /admin (pages) and GET /health.
app.include_router(pages_router)
