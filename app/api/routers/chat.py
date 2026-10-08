import asyncio

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.ai.tools import ChatContext
from app.constants import CHAT_REQUEST_TIMEOUT_SECONDS
from app.dependencies import (
    CookieDependency,
    ParkingGraphDependency,
    ParkingServiceDependency,
    cookie_rate_limit,
    global_user_limit,
)
from app.helpers import set_chat_cookie
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[
        Depends(cookie_rate_limit),
        Depends(global_user_limit),
    ],
)
async def chat(
    payload: ChatRequest,
    response: Response,
    session_id: CookieDependency,
    parking_graph: ParkingGraphDependency,
    parking_service: ParkingServiceDependency,
) -> ChatResponse:
    response.headers["Cache-Control"] = "no-store"

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question must not be blank",
        )

    context = ChatContext(parking_service=parking_service)

    try:
        async with asyncio.timeout(CHAT_REQUEST_TIMEOUT_SECONDS):
            result = await parking_graph.chat(session_id, question, context)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Chat processing timed out. Please retry.",
            headers={"Cache-Control": "no-store"},
        ) from exc

    set_chat_cookie(response, session_id)

    return ChatResponse(status=result["status"], answer=result["answer"], sources=context.sources)
