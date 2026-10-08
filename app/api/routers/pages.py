"""Non-API routes: the chat page, the admin console page, and health check."""

from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from app.config import settings
from app.constants import TEMPLATES_DIR
from app.dependencies import RedisDependency, SQLSessionDependency
from app.helpers import set_chat_cookie

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
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


@router.get("/admin", response_class=HTMLResponse)
async def admin_console(request: Request):
    # The page itself needs no auth to load; it collects the admin key via
    # JS and sends it as X-Admin-Key on each call to the /admin/* API.
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "title": "VilniusPark Admin Console",
        },
    )


@router.get("/health")
async def health(
    db: SQLSessionDependency,
    redis: RedisDependency,
):
    try:
        await db.execute(text("SELECT 1"))
        await redis.ping()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A backing service is unavailable",
        ) from exc

    return {
        "api": "ok",
        "postgres": "ok",
        "redis": "ok",
    }
