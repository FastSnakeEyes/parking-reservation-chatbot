from fastapi import Response

from app.config import settings


def set_chat_cookie(
    response: Response,
    session_id: str,
) -> None:
    response.set_cookie(
        key=settings.cookie_name,
        value=session_id,
        httponly=True,
        secure=settings.app_env.value == "production",
        samesite="strict",
        max_age=settings.cookie_ttl,
        path="/",
    )

    response.headers["Cache-Control"] = "no-store"
