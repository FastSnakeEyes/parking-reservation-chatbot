from dataclasses import dataclass, field
from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from fastapi.encoders import jsonable_encoder
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command
from starlette.concurrency import run_in_threadpool

from app.ai.guardrails import get_pii_guard
from app.config import settings
from app.db.models import Parking
from app.enums import SpaceType
from app.logger import get_logger
from app.schemas.chat import BookingInput
from app.services.parking import ParkingService

logger = get_logger(__name__)


@dataclass
class ChatContext:
    """Per-request dependencies of the user agent (its `context_schema`).

    The agent itself is built once at startup; everything bound to one
    request - the ParkingService over the request-scoped DB session, and the
    sources cited in this turn's answer - arrives here through runtime.context."""

    # A real import, not a TYPE_CHECKING one: building each tool's schema for the
    # model (bind_tools) resolves ToolRuntime[ChatContext] down to this field.
    parking_service: ParkingService
    sources: list[dict] = field(default_factory=list)


def create_parking_tools() -> list[BaseTool]:
    local_timezone = ZoneInfo(settings.timezone)

    @tool
    async def list_parkings(runtime: ToolRuntime[ChatContext]) -> dict:
        """List public parking IDs, names, and districts."""

        result = await runtime.context.parking_service.parking_repository.session.execute(
            sa.select(
                Parking.id,
                Parking.name,
                Parking.district,
            ).order_by(Parking.id)
        )

        return {"parkings": [dict(row) for row in result.mappings().all()]}

    @tool
    async def get_parking_info(
        parking_id: int,
        runtime: ToolRuntime[ChatContext],
        on_date: date | None = None,
    ) -> dict:
        """Read date-specific prices, hours, capacities and restrictions.

        Bookable capacity is not the number of free spaces.
        """

        requested_date = on_date or datetime.now(local_timezone).date()

        info = await runtime.context.parking_service.parking_repository.get_public_info(
            parking_id=parking_id,
            on_date=requested_date,
        )

        if info is None:
            return {"error": "Parking not found"}

        return jsonable_encoder(info)

    @tool
    async def search_public_rules(query: str, runtime: ToolRuntime[ChatContext]) -> dict:
        """Search static public rules and parking information.

        Use public topic keywords only. Do not include customer
        names, car numbers, or private booking details.
        """

        if await run_in_threadpool(get_pii_guard().contains_pii, query):
            return {"error": ("Query must not contain personal details. Rephrase using public topic keywords only.")}

        sources = runtime.context.sources
        documents = await run_in_threadpool(
            runtime.context.parking_service.vector_repository.search,
            query=query,
            k=settings.retriever_k,
        )

        results = []

        for document in documents:
            label = len(sources) + 1
            source = document.metadata.get("source", "unknown")

            sources.append(
                {
                    "label": label,
                    "source": source,
                    "chunk_index": document.metadata.get("chunk_index"),
                }
            )

            results.append(
                {
                    "label": label,
                    "source": source,
                    "text": document.page_content,
                }
            )

        return {"documents": results}

    # return_direct: the review below is the turn's final answer, shown
    # verbatim (exact price/hours/availability) rather than trusting the
    # model's retelling of it.
    @tool(args_schema=BookingInput, return_direct=True)
    async def prepare_reservation(
        parking_id: int,
        space_type: SpaceType,
        first_name: str,
        surname: str,
        car_number: str,
        starts_at: datetime,
        ends_at: datetime,
        runtime: ToolRuntime[ChatContext],
    ) -> Command | dict:
        """Validate complete booking details and prepare a review.

        Does not save or approve a reservation.
        """

        booking = BookingInput(
            parking_id=parking_id,
            space_type=space_type,
            first_name=first_name,
            surname=surname,
            car_number=car_number,
            starts_at=starts_at,
            ends_at=ends_at,
        )

        await runtime.context.parking_service.reservation_repository.validate_request(booking)

        info = await runtime.context.parking_service.parking_repository.get_public_info(
            parking_id=parking_id,
            on_date=booking.starts_at.astimezone(local_timezone).date(),
        )

        if info is None:
            return {"error": "Parking not found"}

        price = info["prices"].get(space_type)

        if price is None:
            tariff = "No applicable tariff found"
        else:
            tariff = (
                f"{price['price_per_hour']} {price['currency']}/hour; "
                f"daily maximum {price['daily_max']} {price['currency']}"
            )

        hours = info["hours"]

        if hours is None:
            schedule = "Opening hours unavailable"
        elif hours["is_closed"]:
            schedule = "Closed on the starting date"
        else:
            schedule = f"{hours['open_time']}–{hours['close_time']}"

        draft = {
            "id": str(uuid4()),
            "data": booking.model_dump(mode="json"),
        }

        review = (
            "Please review your reservation request:\n"
            f"Parking: {info['name']}\n"
            f"Space type: {space_type}\n"
            f"Name: {booking.first_name} {booking.surname}\n"
            f"Car: {booking.car_number}\n"
            f"Start: "
            f"{booking.starts_at.astimezone(local_timezone).isoformat()}\n"
            f"End: "
            f"{booking.ends_at.astimezone(local_timezone).isoformat()}\n"
            f"Starting-date tariff: {tariff}\n"
            f"Starting-date hours: {schedule}\n\n"
            "This is not a total price quote or a guaranteed space. "
            "The request requires administrator approval.\n"
            "Type confirm to submit, cancel to discard, "
            "or describe a correction."
        )

        return Command(
            update={
                "draft": draft,
                "messages": [ToolMessage(content=review, tool_call_id=runtime.tool_call_id)],
            }
        )

    return [
        list_parkings,
        get_parking_info,
        search_public_rules,
        prepare_reservation,
    ]


# --------------------------------------------------------------------------- #
# Admin agent tools
# --------------------------------------------------------------------------- #
@tool
def approve_reservation(summary: str) -> str:
    """Ask the administrator to approve the reservation under review.

    Args:
        summary: Two to four plain sentences describing the reservation for the administrator.
    """
    # Paused by HumanInTheLoopMiddleware until the administrator decides; it only
    # runs once approved. The decision itself is acted on by ParkingGraph's
    # `record` node, not here.
    return "The administrator approved the reservation."
