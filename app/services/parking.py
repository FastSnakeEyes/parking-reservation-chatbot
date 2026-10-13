"""ParkingService: the parking capabilities ParkingGraph's nodes (and the
admin API) call - DB and MCP side effects only. It never calls the graph;
the graph owns the conversation workflow (app/ai/graph.py).
"""

from datetime import datetime
from uuid import UUID
from zoneinfo import ZoneInfo

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings
from app.db.models import Reservation
from app.db.repositories.base import AbstractVectorRepository
from app.db.repositories.parking import ParkingRepository
from app.db.repositories.reservation import ReservationRepository
from app.enums import ReservationStatus
from app.logger import get_logger
from app.schemas.chat import BookingInput

logger = get_logger(__name__)


def _mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "parking": {
                "transport": "streamable_http",
                "url": settings.mcp_server_url,
                "headers": {"Authorization": f"Bearer {settings.mcp_api_key.get_secret_value()}"},
            }
        }
    )


class ParkingService:
    def __init__(
        self,
        parking_repository: ParkingRepository,
        reservation_repository: ReservationRepository,
        vector_repository: AbstractVectorRepository | None = None,
    ):
        """`vector_repository` is only needed by the user agent's tools."""
        self.parking_repository = parking_repository
        self.reservation_repository = reservation_repository
        self.vector_repository = vector_repository
        self.timezone = ZoneInfo(settings.timezone)

    def _period(self, starts_at: datetime, ends_at: datetime) -> str:
        return f"{starts_at.astimezone(self.timezone).isoformat()} - {ends_at.astimezone(self.timezone).isoformat()}"

    async def book(self, draft: dict, conversation_id: str) -> UUID:
        """Save a draft prepared by the prepare_reservation tool as a pending
        reservation. Raises ValueError if the booking is no longer valid
        (e.g. its start has passed)."""
        booking = BookingInput.model_validate(draft["data"])

        reservation_id = await self.reservation_repository.submit_pending(
            reservation_id=UUID(draft["id"]),
            booking=booking,
            conversation_id=conversation_id,
        )

        logger.info(
            "Reservation %s submitted (parking_id=%s, space_type=%s), pending approval",
            reservation_id,
            booking.parking_id,
            booking.space_type,
        )

        return reservation_id

    async def approval_facts(self, reservation_id: UUID) -> dict:
        """The facts the admin agent drafts the approval request from."""
        reservation = await self.reservation_repository.get_by_id(reservation_id)
        info = await self.parking_repository.get_public_info(
            parking_id=reservation.parking_id,
            on_date=reservation.starts_at.astimezone(self.timezone).date(),
        )

        return {
            "Reservation ID": str(reservation_id),
            "Parking": info["name"] if info else str(reservation.parking_id),
            "Space type": reservation.space_type,
            "Customer": f"{reservation.first_name} {reservation.surname}",
            "Car number": reservation.car_number,
            "Period": self._period(reservation.starts_at, reservation.ends_at),
        }

    async def list_pending(self) -> list[Reservation]:
        return await self.reservation_repository.list_pending()

    async def get_pending(self, reservation_id: UUID) -> Reservation | None:
        reservation = await self.reservation_repository.get_by_id(reservation_id)

        if reservation is None or reservation.status != ReservationStatus.PENDING:
            return None

        return reservation

    async def record_decision(
        self, reservation_id: UUID, *, approved: bool, note: str | None = None
    ) -> ReservationStatus | str:
        """Close a pending reservation: on approval, record it through the
        Stage 3 MCP server and mark it RECORDED; otherwise mark it REJECTED.
        Returns the resulting status."""
        reservation = await self.reservation_repository.get_by_id(reservation_id)

        if reservation is None or reservation.status != ReservationStatus.PENDING:
            status = reservation.status if reservation else "missing"
            logger.warning("record_decision: reservation %s not pending (status=%s), skipping", reservation_id, status)
            return status

        if approved:
            approved_at = datetime.now(self.timezone)

            tools = {tool.name: tool for tool in await _mcp_client().get_tools()}
            await tools["record_reservation"].ainvoke(
                {
                    "name": f"{reservation.first_name} {reservation.surname}",
                    "car_number": reservation.car_number,
                    "period": self._period(reservation.starts_at, reservation.ends_at),
                    "approval_time": approved_at.isoformat(),
                }
            )

            await self.reservation_repository.update(
                reservation_id, {"status": ReservationStatus.RECORDED, "approved_at": approved_at}
            )
            logger.info("Reservation %s approved and recorded via MCP", reservation_id)
            status = ReservationStatus.RECORDED
        else:
            await self.reservation_repository.update(reservation_id, {"status": ReservationStatus.REJECTED})
            logger.info("Reservation %s rejected (note=%s)", reservation_id, note)
            status = ReservationStatus.REJECTED

        await self.reservation_repository.session.commit()

        return status
