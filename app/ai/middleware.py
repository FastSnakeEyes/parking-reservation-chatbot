"""Deterministic, model-free steps of a chat turn, run as the user agent's
before_agent middleware - ahead of the model, ending the turn early when
they produce the answer themselves.

Order of checks (one middleware, so the order is explicit here):
1. a pending administrator decision on the customer's last reservation
2. prompt injection
3. the `cancel` / `confirm` commands - the model cannot authorize a DB write
4. otherwise: drop the previous review's draft, then let the model run
"""

from typing import NotRequired
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain.agents import AgentState
from langchain.agents.middleware import before_agent
from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from starlette.concurrency import run_in_threadpool

from app.ai.guardrails import is_prompt_injection
from app.ai.tools import ChatContext
from app.config import settings
from app.db.repositories.reservation import ReservationRepository
from app.logger import get_logger
from app.schemas.chat import BookingInput

logger = get_logger(__name__)


class ChatState(AgentState):
    draft: NotRequired[dict | None]  # set by prepare_reservation, consumed by `confirm`
    last_reservation_id: NotRequired[str | None]  # for the one-time decision notice


def _answer(text: str, *, drop: str | None, **updates) -> dict:
    """End the turn with `text`. `drop` removes the triggering human message
    (or the whole conversation, with REMOVE_ALL_MESSAGES) from the history."""
    messages = [RemoveMessage(id=drop)] if drop else []
    return {"messages": [*messages, AIMessage(content=text)], "jump_to": "end", **updates}


async def decision_notice(reservations: ReservationRepository, reservation_id: str | None) -> str | None:
    """A one-time notice once the administrator has decided on the customer's
    last reservation - otherwise the customer never finds out the outcome."""
    if not reservation_id:
        return None

    reservation = await reservations.get_by_id(UUID(reservation_id))

    if reservation is None or reservation.status not in ("recorded", "rejected"):
        return None

    outcome = "approved and recorded" if reservation.status == "recorded" else "rejected by the administrator"
    logger.info("Delivering decision notice for reservation %s: %s", reservation_id, reservation.status)

    return f"Update: your reservation {reservation_id} was {outcome}.\nYou can ask your question again."


async def _start_approval(context: ChatContext, reservation_id: UUID, booking: BookingInput) -> None:
    timezone = ZoneInfo(settings.timezone)
    info = await context.repository.get_public_info(
        parking_id=booking.parking_id,
        on_date=booking.starts_at.astimezone(timezone).date(),
    )
    period = (
        f"{booking.starts_at.astimezone(timezone).isoformat()} - {booking.ends_at.astimezone(timezone).isoformat()}"
    )

    await context.approval_pipeline.start(
        reservation_id,
        {
            "Reservation ID": str(reservation_id),
            "Parking": info["name"] if info else str(booking.parking_id),
            "Space type": booking.space_type,
            "Customer": f"{booking.first_name} {booking.surname}",
            "Car number": booking.car_number,
            "Period": period,
        },
    )


async def _confirm(state: ChatState, context: ChatContext, question_id: str) -> dict:
    draft = state.get("draft")

    if draft is None:
        previous_id = state.get("last_reservation_id")
        if previous_id:
            return _answer(f"Your previous request was already submitted.\nReference: {previous_id}", drop=question_id)
        return _answer("There is no reviewed reservation to confirm.", drop=question_id)

    booking = BookingInput.model_validate(draft["data"])

    try:
        reservation_id = await context.reservations.submit_pending(reservation_id=UUID(draft["id"]), booking=booking)
    except ValueError as exc:
        return _answer(
            f"The request cannot be submitted: {exc}. Please provide corrected details.",
            drop=question_id,
            draft=None,
        )

    logger.info(
        "Reservation %s submitted (parking_id=%s, space_type=%s), pending approval",
        reservation_id,
        booking.parking_id,
        booking.space_type,
    )

    # Stage 2/4: hand the reservation off to the admin-approval graph.
    await _start_approval(context, reservation_id, booking)

    # Start fresh after a submitted booking.
    return _answer(
        f"Your reservation request was saved and is awaiting administrator approval.\nReference: {reservation_id}",
        drop=REMOVE_ALL_MESSAGES,
        draft=None,
        last_reservation_id=str(reservation_id),
    )


@before_agent(state_schema=ChatState, can_jump_to=["end"], name="SessionCommands")
async def session_commands(state: ChatState, runtime: Runtime[ChatContext]) -> dict | None:
    question = state["messages"][-1]
    text = str(question.content).strip()

    notice = await decision_notice(runtime.context.reservations, state.get("last_reservation_id"))
    if notice is not None:
        return _answer(notice, drop=question.id, last_reservation_id=None)

    if await run_in_threadpool(is_prompt_injection, text):
        logger.warning("Blocked a suspected prompt injection")
        # Dropped from the history too, so it never reaches the model on a later turn.
        return _answer(
            "I can't process that request. Please ask about parking information or reservations.",
            drop=question.id,
        )

    command = text.casefold()

    if command == "cancel":
        return _answer("The conversation and unsent booking draft were cleared.", drop=REMOVE_ALL_MESSAGES, draft=None)

    if command == "confirm":
        return await _confirm(state, runtime.context, question.id)

    # Invalidate an old review before processing a correction.
    return {"draft": None}
