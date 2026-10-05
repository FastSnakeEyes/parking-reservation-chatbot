import json
from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from redis.asyncio import Redis


class RedisChatHistory(BaseChatMessageHistory):
    """Async chat history in a Redis list, capped to the last `max_messages` messages."""

    def __init__(self, redis: Redis, session_id: str, ttl: int, max_messages: int):
        self.redis = redis
        self.key = f"chat:history:{session_id}"
        self.ttl = ttl
        self.max_messages = max_messages

    @staticmethod
    def _dump(messages: Sequence[BaseMessage]) -> list[str]:
        return [json.dumps(message_to_dict(m), ensure_ascii=False) for m in messages]

    async def aget_messages(self) -> list[BaseMessage]:
        raw = await self.redis.lrange(self.key, 0, -1)
        return messages_from_dict([json.loads(item) for item in raw])

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.rpush(self.key, *self._dump(messages))
            pipe.ltrim(self.key, -self.max_messages, -1)  # keep only the newest messages
            pipe.expire(self.key, self.ttl)               # expires together with the cookie
            await pipe.execute()

    async def areplace(self, messages: Sequence[BaseMessage]) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(self.key)
            if messages:
                pipe.rpush(self.key, *self._dump(messages))
                pipe.expire(self.key, self.ttl)
            await pipe.execute()

    async def aclear(self) -> None:
        await self.redis.delete(self.key)

    # Sync API of the base class: this app is async-only.
    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        raise NotImplementedError("Use aget_messages()")

    def clear(self) -> None:
        raise NotImplementedError("Use aclear()")