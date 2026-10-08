from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routers.admin import router as admin_router
from app.api.routers.chat import router as chat_router
from app.api.routers.pages import router as pages_router
from app.constants import STATIC_DIR
from app.lifespan import lifespan
from app.logger import setup_logging

setup_logging()

app = FastAPI(
    title="Parking chatbot",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)

# Defines POST /chat using ParkingService.ask (the user agent).
app.include_router(chat_router)

# Defines the admin console API (Stage 2); the /admin page and
# ui/admin_app.py (Streamlit) are both clients of it.
app.include_router(admin_router)

# Defines GET /, GET /admin (pages) and GET /health.
app.include_router(pages_router)
