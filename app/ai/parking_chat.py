import asyncio

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.graph import ApprovalPipeline
from app.ai.tools import ChatContext
from app.db.repositories.base import AbstractVectorRepository
from app.db.repositories.parking import ParkingRepository
from app.db.repositories.reservation import ReservationRepository
from app.logger import get_logger

logger = get_logger(__name__)

REQUEST_TIMEOUT_SECONDS = 120

# The approval graph shares the checkpointer, keyed by reservation_id; this
# prefix keeps chat threads apart (and lets app/ai/chat_cleanup.py find them).
CHAT_THREAD_PREFIX = "chat:"


class ParkingChat:
    """One chat turn: lock the session, run the user agent (app/ai/agents.py)
    on its checkpointer thread, return the answer. Memory, commands and
    guardrails live in the agent itself."""

    def __init__(
        self,
        session: AsyncSession,
        redis: Redis,
        vector_repository: AbstractVectorRepository,
        agent: CompiledStateGraph,
        approval_pipeline: ApprovalPipeline,
    ):
        self.redis = redis
        self.agent = agent
        self.vector_repository = vector_repository
        self.repository = ParkingRepository(session)
        self.reservations = ReservationRepository(session)
        self.approval_pipeline = approval_pipeline

    @staticmethod
    def response(answer: str, sources: list | None = None) -> dict:
        return {
            "answer": answer,
            "sources": sources or [],
        }

    @staticmethod
    def thread_id(session_id: str) -> str:
        return f"{CHAT_THREAD_PREFIX}{session_id}"

    async def ask(self, session_id: str, question: str) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Question must not be blank")

        lock = self.redis.lock(f"parking:chat:{session_id}:lock", timeout=180, blocking_timeout=1)

        if not await lock.acquire():
            return self.response("Your previous message is still being processed.")

        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                return await self._ask(session_id, question)
        finally:
            await lock.release()

    async def _ask(self, session_id: str, question: str) -> dict:
        context = ChatContext(
            repository=self.repository,
            vector_repository=self.vector_repository,
            reservations=self.reservations,
            approval_pipeline=self.approval_pipeline,
        )

        try:
            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": self.thread_id(session_id)}},
                context=context,
                durability="exit",  # one checkpoint per turn, not one per step
            )
        except GraphRecursionError:
            return self.response("I could not finish processing that request. Please try a more specific question.")

        return self.response(self.extract_text(result["messages"][-1].content), context.sources)

    @staticmethod
    def extract_text(content) -> str:
        if isinstance(content, str):
            return content

        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
