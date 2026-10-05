from contextlib import AsyncExitStack, asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.concurrency import run_in_threadpool

from app.ai.graph import ApprovalPipeline
from app.ai.guardrails import get_pii_guard
from app.api.routers.chat import router as chat_router
from app.config import settings
from app.dependencies import (
    ParkingRepositoryDependency,
    SQLSessionDependency,
    create_vector_repository,
)
from app.helpers import set_chat_cookie

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

        Path(settings.graph_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(settings.graph_checkpoint_path)
        )
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

templates = Jinja2Templates(
    directory=str(APP_DIR / "templates"),
)

# Defines POST /chat using ParkingChat and SQL/vector tools.
app.include_router(chat_router)


async def get_redis(request: Request) -> Redis:
    return request.app.state.redis


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    response = templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "title": "Vilnius Parking Assistant",
        },
    )

    cookie = request.cookies.get(settings.cookie_name)

    try:
        session_id = str(UUID(cookie)) if cookie else str(uuid4())
    except ValueError:
        session_id = str(uuid4())

    set_chat_cookie(response, session_id)

    return response


@app.get("/health")
async def health(
    db: SQLSessionDependency,
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

    return {
        "api": "ok",
        "postgres": "ok",
        "redis": "ok",
    }


@app.get("/parkings")
async def list_parkings(
    repository: ParkingRepositoryDependency,
    offset: int = 0,
    limit: int = 100,
):
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(
            status_code=422,
            detail="offset must be >= 0 and limit must be between 1 and 100",
        )

    parkings = await repository.list(
        offset=offset,
        limit=limit,
    )

    return [
        {
            "id": parking.id,
            "slug": parking.slug,
            "name": parking.name,
            "district": parking.district,
            "latitude": parking.latitude,
            "longitude": parking.longitude,
            "total_spaces": parking.total_spaces,
            "bookable_spaces": parking.bookable_spaces,
        }
        for parking in parkings
    ]


@app.get("/parkings/{parking_id}")
async def parking_info(
    parking_id: int,
    repository: ParkingRepositoryDependency,
    on_date: date | None = None,
):
    requested_date = on_date or datetime.now(ZoneInfo(settings.timezone)).date()

    info = await repository.get_public_info(
        parking_id=parking_id,
        on_date=requested_date,
    )

    if info is None:
        raise HTTPException(
            status_code=404,
            detail="Parking not found",
        )

    return info
