"""The chat endpoint's rate limits (app/dependencies.py)."""

from types import SimpleNamespace

import pytest
from app.config import settings
from app.dependencies import cookie_rate_limit, global_user_limit
from fastapi import HTTPException


class LimitsRedis:
    """In-memory stand-in for the Redis counters and hash the limits use."""

    def __init__(self):
        self.counters: dict[str, int] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    async def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def expire(self, key, seconds):
        return True

    async def ttl(self, key):
        return 7

    async def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    async def hdel(self, key, *fields):
        for field in fields:
            self.hashes.get(key, {}).pop(field, None)

    async def hexists(self, key, field):
        return field in self.hashes.get(key, {})

    async def hlen(self, key):
        return len(self.hashes.get(key, {}))

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value


@pytest.fixture
def request_with_redis():
    redis = LimitsRedis()
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(redis=redis))), redis


@pytest.mark.asyncio
async def test_cookie_rate_limit_allows_the_limit_then_refuses(request_with_redis, monkeypatch):
    request, _ = request_with_redis
    monkeypatch.setattr(settings, "rate_limit_times", 2)

    await cookie_rate_limit(request, "user-a")
    await cookie_rate_limit(request, "user-a")
    with pytest.raises(HTTPException) as refused:
        await cookie_rate_limit(request, "user-a")

    assert refused.value.status_code == 429
    assert refused.value.headers["Retry-After"] == "7"
    await cookie_rate_limit(request, "user-b")  # limits are per user


@pytest.mark.asyncio
async def test_global_user_limit_refuses_a_new_user_when_full(request_with_redis, monkeypatch):
    request, _ = request_with_redis
    monkeypatch.setattr(settings, "max_active_users", 2)

    await global_user_limit(request, "user-a")
    await global_user_limit(request, "user-b")
    with pytest.raises(HTTPException) as refused:
        await global_user_limit(request, "user-c")

    assert refused.value.status_code == 429
    await global_user_limit(request, "user-a")  # an already active user is never refused


@pytest.mark.asyncio
async def test_global_user_limit_frees_slots_of_idle_users(request_with_redis, monkeypatch):
    request, redis = request_with_redis
    monkeypatch.setattr(settings, "max_active_users", 1)
    redis.hashes["active_users"] = {"idle-user": "0"}  # last seen at the epoch

    await global_user_limit(request, "user-a")

    assert set(redis.hashes["active_users"]) == {"user-a"}
