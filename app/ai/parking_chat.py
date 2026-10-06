import asyncio
import json
from datetime import datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi.encoders import jsonable_encoder
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.ai.admin_agent import draft_notification
from app.ai.chat_memory import SummaryBufferMemory
from app.ai.chat_store import ChatStore
from app.ai.graph import ApprovalPipeline
from app.ai.guardrails import get_pii_guard, is_prompt_injection
from app.ai.prompts import BUILD_PROMPT
from app.ai.tools import ToolContext, create_parking_tools
from app.config import settings
from app.db.repositories.base import AbstractVectorRepository
from app.db.repositories.parking import ParkingRepository
from app.db.repositories.reservation import ReservationRepository
from app.logger import get_logger
from app.schemas.chat import BookingInput

logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 120


class ParkingChat:
    def __init__(
        self,
        session: AsyncSession,
        redis: Redis,
        vector_repository: AbstractVectorRepository,
        llm: ChatOpenAI,
        approval_pipeline: ApprovalPipeline,
        summary_llm: ChatOpenAI | None = None,
    ):
        self.llm = llm
        self.summary_llm = summary_llm or llm
        self.redis = redis
        self.vector_repository = vector_repository
        self.repository = ParkingRepository(session)
        self.reservations = ReservationRepository(session)
        self.store = ChatStore(redis)
        self.approval_pipeline = approval_pipeline
        self.timezone = ZoneInfo(settings.timezone)

    @staticmethod
    def response(answer: str, sources: list | None = None) -> dict:
        return {
            "answer": answer,
            "sources": sources or [],
        }

    def memory(self, session_id: str) -> SummaryBufferMemory:
        return SummaryBufferMemory(
            self.redis,
            session_id,
            summarizer=self.summary_llm,
        )

    async def confirm(self, state: dict, memory: SummaryBufferMemory) -> dict:
        draft = state.get("draft")

        if draft is None:
            previous_id = state.get("last_reservation_id")

            if previous_id:
                return self.response(f"Your previous request was already submitted.\nReference: {previous_id}")

            return self.response("There is no reviewed reservation to confirm.")

        booking = BookingInput.model_validate(draft["data"])

        try:
            reservation_id = await self.reservations.submit_pending(
                reservation_id=UUID(draft["id"]),
                booking=booking,
            )
        except ValueError as exc:
            state["draft"] = None

            return self.response(f"The request cannot be submitted: {exc}. Please provide corrected details.")

        state["draft"] = None
        state["last_reservation_id"] = str(reservation_id)
        await memory.clear()  # start fresh after a submitted booking

        logger.info(
            "Reservation %s submitted (parking_id=%s, space_type=%s), pending approval",
            reservation_id,
            booking.parking_id,
            booking.space_type,
        )

        # Stage 2/4: hand the reservation off to the admin-approval graph.
        await self.start_approval(reservation_id, booking)

        return self.response(
            f"Your reservation request was saved and is awaiting administrator approval.\nReference: {reservation_id}"
        )

    async def start_approval(self, reservation_id: UUID, booking: BookingInput) -> None:
        info = await self.repository.get_public_info(
            parking_id=booking.parking_id,
            on_date=booking.starts_at.astimezone(self.timezone).date(),
        )
        period = (
            f"{booking.starts_at.astimezone(self.timezone).isoformat()} - "
            f"{booking.ends_at.astimezone(self.timezone).isoformat()}"
        )

        message = await draft_notification(
            self.llm,
            {
                "Reservation ID": str(reservation_id),
                "Parking": info["name"] if info else str(booking.parking_id),
                "Space type": booking.space_type,
                "Customer": f"{booking.first_name} {booking.surname}",
                "Car number": booking.car_number,
                "Period": period,
            },
        )

        await self.approval_pipeline.start(reservation_id, message)

    async def pending_decision_notice(self, state: dict) -> str | None:
        """A one-time proactive notice when the administrator has decided on
        the customer's last reservation since they were last in this
        session - otherwise the customer never finds out the outcome."""
        reservation_id = state.get("last_reservation_id")

        if not reservation_id:
            return None

        reservation = await self.reservations.get_by_id(UUID(reservation_id))

        if reservation is None or reservation.status not in ("recorded", "rejected"):
            return None

        state["last_reservation_id"] = None  # deliver this notice only once

        outcome = "approved and recorded" if reservation.status == "recorded" else "rejected by the administrator"
        logger.info("Delivering decision notice for reservation %s: %s", reservation_id, reservation.status)

        return f"Update: your reservation {reservation_id} was {outcome}.\nYou can ask your question again."

    async def ask(self, session_id: str, question: str) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Question must not be blank")

        lock = self.store.lock(session_id)

        if not await lock.acquire():
            return self.response("Your previous message is still being processed.")

        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                return await self._ask(session_id, question)
        finally:
            await lock.release()

    async def _ask(self, session_id: str, question: str) -> dict:
        if await run_in_threadpool(is_prompt_injection, question):
            logger.warning("Blocked a suspected prompt injection (session=%s)", session_id)
            return self.response("I can't process that request. Please ask about parking information or reservations.")

        state = await self.store.load(session_id)

        notice = await self.pending_decision_notice(state)

        if notice is not None:
            await self.store.save(session_id, state)
            return self.response(notice)

        memory = self.memory(session_id)
        command = question.casefold()

        if command == "cancel":
            await self.store.clear(session_id)
            await memory.clear()

            return self.response("The conversation and unsent booking draft were cleared.")

        # The model cannot authorize a database write.
        if command == "confirm":
            result = await self.confirm(state, memory)
            await self.store.save(session_id, state)
            return result

        # Invalidate an old review before processing a correction.
        state["draft"] = None
        await self.store.save(session_id, state)

        context = ToolContext()

        tools = create_parking_tools(
            repository=self.repository,
            vector_repository=self.vector_repository,
            reservations=self.reservations,
            context=context,
        )
        tools_by_name = {item.name: item for item in tools}

        model = self.llm.bind_tools(
            tools,
            parallel_tool_calls=False,
        )

        prompt = await BUILD_PROMPT.ainvoke(
            {
                **await memory.load(),  # running summary + recent turns
                "question": question,
                "now": datetime.now(self.timezone).strftime("%Y-%m-%d %H:%M"),
                "timezone": settings.timezone,
            }
        )
        messages = prompt.to_messages()

        for _ in range(settings.max_tool_rounds):
            answer = await model.ainvoke(messages)
            messages.append(answer)

            if not answer.tool_calls:
                text = self.extract_text(answer.content)
                # Defense in depth: redact any PII surfaced by retrieval or the
                # model itself. The booking review (context.review) is handled
                # separately below since it legitimately echoes the user's own
                # just-supplied name/car number back to them.
                text = await run_in_threadpool(get_pii_guard().redact, text)

                # Store only question + final answer; tool chatter stays in this turn.
                await memory.add_turn(
                    [
                        HumanMessage(content=question),
                        AIMessage(content=text),
                    ]
                )

                return self.response(text, context.sources)

            # Sequential execution for the request's SQL session.
            for call in answer.tool_calls:
                selected_tool = tools_by_name.get(call["name"])

                if context.review is not None:
                    result = {"error": "Review ready; wait for confirmation."}
                elif selected_tool is None:
                    result = {"error": "Unknown tool"}
                else:
                    try:
                        result = await selected_tool.ainvoke(call["args"])
                    except (ValidationError, ValueError):
                        result = {"error": ("Invalid arguments. Ask the user to clarify the required details.")}

                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            jsonable_encoder(result),
                            ensure_ascii=False,
                        ),
                        tool_call_id=call["id"],
                    )
                )

            if context.review is not None:
                state["draft"] = context.draft
                await self.store.save(session_id, state)

                await memory.add_turn(
                    [
                        HumanMessage(content=question),
                        AIMessage(content=context.review),
                    ]
                )

                return self.response(context.review, context.sources)

        return self.response("I could not finish processing that request. Please try a more specific question.")

    @staticmethod
    def extract_text(content) -> str:
        if isinstance(content, str):
            return content

        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
