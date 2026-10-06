import json
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from redis.asyncio import Redis


class RedisChatHistory:
    """Async chat history in a Redis list, capped to the last `max_messages` messages.

    Plain class, not a langchain_core.chat_history.BaseChatMessageHistory
    subclass: this app is async-only, and RedisChatHistory is only ever
    called directly (by SummaryBufferMemory), never passed into a LangChain
    utility that would dispatch through that base class's interface - so
    there's no reason to take on its sync-vs-async `a`-prefixed method
    names (aadd_messages, aget_messages, ...).
    """

    def __init__(self, redis: Redis, session_id: str, ttl: int, max_messages: int):
        self.redis = redis
        self.key = f"chat:history:{session_id}"
        self.ttl = ttl
        self.max_messages = max_messages

    @staticmethod
    def _dump(messages: Sequence[BaseMessage]) -> list[str]:
        return [json.dumps(message_to_dict(m), ensure_ascii=False) for m in messages]

    async def get_messages(self) -> list[BaseMessage]:
        raw = await self.redis.lrange(self.key, 0, -1)
        return messages_from_dict([json.loads(item) for item in raw])

    async def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.rpush(self.key, *self._dump(messages))
            pipe.ltrim(self.key, -self.max_messages, -1)  # keep only the newest messages
            pipe.expire(self.key, self.ttl)  # expires together with the cookie
            await pipe.execute()

    async def replace(self, messages: Sequence[BaseMessage]) -> None:
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(self.key)
            if messages:
                pipe.rpush(self.key, *self._dump(messages))
                pipe.expire(self.key, self.ttl)
            await pipe.execute()

    async def clear(self) -> None:
        await self.redis.delete(self.key)
