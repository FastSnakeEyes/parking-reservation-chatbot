from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, get_buffer_string, trim_messages
from langchain_core.output_parsers import StrOutputParser
from redis.asyncio import Redis

from app.ai.prompts import SUMMARY_PROMPT
from app.ai.token_counting import count_tokens
from app.config import settings
from app.ai.chat_history import RedisChatHistory


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return value or ""


class SummaryBufferMemory:
    """Recent messages verbatim + a running LLM summary of older ones.

    Same idea as the legacy ConversationSummaryBufferMemory, but async,
    Redis-backed and built from current LangChain classes.
    """

    def __init__(self, redis: Redis, session_id: str, summarizer: BaseChatModel):
        self.redis = redis
        self.history = RedisChatHistory(
            redis,
            session_id,
            ttl=settings.cookie_ttl,
            max_messages=settings.history_max_stored,
        )
        self.summary_key = f"chat:summary:{session_id}"
        self.summarize = SUMMARY_PROMPT | summarizer | StrOutputParser()

    async def load(self) -> dict:
        """Prompt variables: {'summary': [SystemMessage] or [], 'history': [...]}."""
        summary = _text(await self.redis.get(self.summary_key))
        return {
            "summary": (
                [SystemMessage(content=f"Summary of the earlier conversation:\n{summary}")]
                if summary
                else []
            ),
            "history": await self.history.aget_messages(),
        }

    async def add_turn(self, messages: list[BaseMessage]) -> None:
        await self.history.aadd_messages(messages)
        await self._prune()

    async def clear(self) -> None:
        await self.history.aclear()
        await self.redis.delete(self.summary_key)

    async def _prune(self) -> None:
        """When history exceeds history_max_tokens, fold the oldest turns into the summary."""
        messages = await self.history.aget_messages()
        if count_tokens(messages) <= settings.history_max_tokens:
            return

        recent = trim_messages(
            messages,
            max_tokens=settings.history_keep_tokens,
            strategy="last",
            token_counter=count_tokens,
            start_on="human",
        )
        older = messages[: len(messages) - len(recent)]
        if not older:
            return

        current = _text(await self.redis.get(self.summary_key))
        new_summary = await self.summarize.ainvoke({
            "summary": current or "(empty)",
            "new_lines": get_buffer_string(older),
        })

        await self.redis.set(self.summary_key, new_summary, ex=settings.cookie_ttl)
        await self.history.areplace(recent)