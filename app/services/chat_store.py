import json

from redis.asyncio import Redis


class ChatStore:
    TTL_SECONDS = 1800

    def __init__(self, redis: Redis):
        self.redis = redis

    @staticmethod
    def key(session_id: str) -> str:
        return f"parking:chat:{session_id}"

    async def load(self, session_id: str) -> dict:
        raw = await self.redis.get(self.key(session_id))

        if raw is None:
            return {
                "history": [],
                "draft": None,
            }

        return json.loads(raw)

    async def save(self, session_id: str, state: dict) -> None:
        await self.redis.set(
            self.key(session_id),
            json.dumps(state),
            ex=self.TTL_SECONDS,
        )

    async def clear(self, session_id: str) -> None:
        await self.redis.delete(self.key(session_id))

    def lock(self, session_id: str):
        return self.redis.lock(
            f"{self.key(session_id)}:lock",
            timeout=180,
            blocking_timeout=1,
        )
