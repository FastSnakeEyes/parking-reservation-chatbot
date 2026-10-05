import pytest
from app.ai.chat_memory import SummaryBufferMemory
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture
def memory(fake_redis):
    return SummaryBufferMemory(
        fake_redis,
        "test-session",
        summarizer=FakeListChatModel(responses=["summary"]),
    )


@pytest.mark.asyncio
async def test_add_turn_then_load_round_trips(memory):
    await memory.add_turn(
        [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ]
    )

    loaded = await memory.load()

    assert loaded["summary"] == []
    assert [message.content for message in loaded["history"]] == ["hello", "hi there"]


@pytest.mark.asyncio
async def test_clear_empties_history_and_summary(memory):
    await memory.add_turn([HumanMessage(content="hello")])
    await memory.clear()

    loaded = await memory.load()

    assert loaded["history"] == []
    assert loaded["summary"] == []


@pytest.mark.asyncio
async def test_separate_sessions_do_not_share_history(fake_redis):
    first = SummaryBufferMemory(fake_redis, "session-a", summarizer=FakeListChatModel(responses=["s"]))
    second = SummaryBufferMemory(fake_redis, "session-b", summarizer=FakeListChatModel(responses=["s"]))

    await first.add_turn([HumanMessage(content="only in session a")])

    assert (await second.load())["history"] == []
