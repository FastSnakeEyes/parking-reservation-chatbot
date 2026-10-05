from app.ai.token_counting import count_tokens, get_token_counter
from app.config import settings
from langchain_core.messages import HumanMessage


def test_approximate_counter_counts_nonempty_messages(monkeypatch):
    monkeypatch.setattr(settings, "token_counter", "approximate")
    get_token_counter.cache_clear()

    try:
        tokens = count_tokens([HumanMessage(content="hello there, how are you?")])
    finally:
        get_token_counter.cache_clear()

    assert tokens > 0


def test_counter_is_cached_per_settings(monkeypatch):
    monkeypatch.setattr(settings, "token_counter", "approximate")
    get_token_counter.cache_clear()

    try:
        assert get_token_counter() is get_token_counter()
    finally:
        get_token_counter.cache_clear()


def test_empty_messages_counts_to_zero_or_more(monkeypatch):
    monkeypatch.setattr(settings, "token_counter", "approximate")
    get_token_counter.cache_clear()

    try:
        assert count_tokens([]) >= 0
    finally:
        get_token_counter.cache_clear()
