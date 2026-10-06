"""ParkingInformationService: read-only parking data for ChatAgent.

This is the only way agent code should touch parking data - no raw SQL, no
direct repository access from app/ai/*. Not wired into app/ai/tools.py yet
(that's Phase 3); this just builds the service it will be routed through.
"""

from datetime import date

from langchain_core.documents import Document
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import AbstractVectorRepository
from app.db.repositories.parking import ParkingRepository


class ParkingInformationService:
    def __init__(self, session: AsyncSession, vector_repository: AbstractVectorRepository):
        self.repository = ParkingRepository(session)
        self.vector_repository = vector_repository

    async def list_parkings(self) -> list[dict]:
        parkings = await self.repository.list(limit=100)

        return [{"id": parking.id, "name": parking.name, "district": parking.district} for parking in parkings]

    async def get_parking_info(self, parking_id: int, on_date: date) -> dict | None:
        return await self.repository.get_public_info(parking_id=parking_id, on_date=on_date)

    def search_public_rules(self, query: str, k: int) -> list[Document]:
        return self.vector_repository.search(query=query, k=k)
