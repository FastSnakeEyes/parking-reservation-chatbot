"""ReservationExportService: invokes the restricted MCP export tool for
confirmed reservations only, and tracks export_status separately from the
reservation's own business status - so an export failure can never lose or
revert an already-confirmed ("recorded") booking. Not wired into the live
approval path yet (Phase 5); built and tested now against an injected
MCP client factory.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.repositories.reservation import ReservationRepository


class MCPTool(Protocol):
    async def ainvoke(self, args: dict) -> object: ...


class MCPClient(Protocol):
    async def get_tools(self) -> list[MCPTool]: ...


class ReservationExportService:
    def __init__(self, session: AsyncSession, mcp_client_factory: Callable[[], MCPClient]):
        self.session = session
        self.reservations = ReservationRepository(session)
        self._mcp_client_factory = mcp_client_factory
        self.timezone = ZoneInfo(settings.timezone)

    async def export(self, reservation_id: UUID) -> bool:
        """Returns True once export_status='exported' - whether that
        happened just now or on a previous call (idempotent retry).

        Locks the reservation row for the duration of the call (including
        the MCP network call) so two concurrent export() calls for the same
        reservation can't both pass the export_status check before either
        commits and double-write. Holding a row lock across a network call
        is usually worth avoiding, but export is low-frequency and
        per-reservation, so the simplicity is worth it here; Phase 5 adds a
        second, independent layer of protection (dedup by reservation_id on
        the MCP server itself).
        """
        reservation = await self.reservations.get_for_update(reservation_id)

        if reservation is None or reservation.status != "recorded":
            # Not authoritatively confirmed - nothing to export.
            return False

        if reservation.export_status == "exported":
            return True

        period = (
            f"{reservation.starts_at.astimezone(self.timezone).isoformat()} - "
            f"{reservation.ends_at.astimezone(self.timezone).isoformat()}"
        )
        approval_time = reservation.approved_at.astimezone(self.timezone).isoformat() if reservation.approved_at else ""

        try:
            client = self._mcp_client_factory()
            tools = {tool.name: tool for tool in await client.get_tools()}
            await tools["record_reservation"].ainvoke(
                {
                    "name": f"{reservation.first_name} {reservation.surname}",
                    "car_number": reservation.car_number,
                    "period": period,
                    "approval_time": approval_time,
                }
            )
        except Exception:
            reservation.export_status = "failed"
            await self.session.commit()
            return False

        reservation.export_status = "exported"
        reservation.exported_at = datetime.now(UTC)
        await self.session.commit()

        return True
