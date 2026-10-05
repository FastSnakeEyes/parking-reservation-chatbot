from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from fastapi.encoders import jsonable_encoder
from langchain_core.tools import BaseTool, tool
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.db.models import Parking
from app.db.repositories.base import AbstractVectorRepository
from app.db.repositories.parking import ParkingRepository
from app.schemas.chat import BookingInput
from app.db.repositories.base import AbstractSQLRepository


@dataclass
class ToolContext:
    sources: list[dict] = field(default_factory=list)
    draft: dict | None = None
    review: str | None = None


def create_parking_tools(
    repository: ParkingRepository,
    vector_repository: AbstractVectorRepository,
    reservations: AbstractSQLRepository,
    context: ToolContext,
) -> list[BaseTool]:
    local_timezone = ZoneInfo(settings.timezone)

    @tool
    async def list_parkings() -> dict:
        """List public parking IDs, names, and districts."""

        result = await repository.session.execute(
            sa.select(
                Parking.id,
                Parking.name,
                Parking.district,
            ).order_by(Parking.id)
        )

        return {
            "parkings": [
                dict(row)
                for row in result.mappings().all()
            ]
        }

    @tool
    async def get_parking_info(
        parking_id: int,
        on_date: date | None = None,
    ) -> dict:
        """Read date-specific prices, hours, capacities and restrictions.

        Bookable capacity is not the number of free spaces.
        """

        requested_date = on_date or datetime.now(
            local_timezone
        ).date()

        info = await repository.get_public_info(
            parking_id=parking_id,
            on_date=requested_date,
        )

        if info is None:
            return {"error": "Parking not found"}

        return jsonable_encoder(info)

    @tool
    async def search_public_rules(query: str) -> dict:
        """Search static public rules and parking information.

        Use public topic keywords only. Do not include customer
        names, car numbers, or private booking details.
        """

        documents = await run_in_threadpool(
            vector_repository.search,
            query=query,
            k=settings.retriever_k,
        )

        results = []

        for document in documents:
            label = len(context.sources) + 1
            source = document.metadata.get("source", "unknown")

            context.sources.append({
                "label": label,
                "source": source,
                "chunk_index": document.metadata.get("chunk_index"),
            })

            results.append({
                "label": label,
                "source": source,
                "text": document.page_content,
            })

        return {"documents": results}

    @tool(args_schema=BookingInput)
    async def prepare_reservation(
        parking_id: int,
        space_type: Literal["standard", "ev"],
        first_name: str,
        surname: str,
        car_number: str,
        starts_at: datetime,
        ends_at: datetime,
    ) -> dict:
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

        await reservations.validate_request(booking)

        info = await repository.get_public_info(
            parking_id=parking_id,
            on_date=booking.starts_at.astimezone(
                local_timezone
            ).date(),
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
            schedule = (
                f"{hours['open_time']}–{hours['close_time']}"
            )

        context.draft = {
            "id": str(uuid4()),
            "data": booking.model_dump(mode="json"),
        }

        context.review = (
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

        return {"review_ready": True}

    return [
        list_parkings,
        get_parking_info,
        search_public_rules,
        prepare_reservation,
    ]