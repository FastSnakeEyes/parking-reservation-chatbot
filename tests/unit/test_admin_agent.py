import pytest
from app.ai.admin_agent import draft_notification
from langchain_core.language_models.fake_chat_models import FakeListChatModel


@pytest.mark.asyncio
async def test_draft_notification_returns_llm_output():
    llm = FakeListChatModel(responses=["Please approve John Doe's reservation."])

    message = await draft_notification(llm, {"Customer": "John Doe", "Car number": "ABC123"})

    assert message == "Please approve John Doe's reservation."


@pytest.mark.asyncio
async def test_draft_notification_feeds_all_details_to_the_prompt():
    captured: list[str] = []

    class RecordingFakeChatModel(FakeListChatModel):
        async def ainvoke(self, input, *args, **kwargs):
            captured.append(str(input))
            return await super().ainvoke(input, *args, **kwargs)

    llm = RecordingFakeChatModel(responses=["ok"])

    await draft_notification(llm, {"Reservation ID": "abc-123", "Car number": "XYZ999"})

    assert any("abc-123" in text and "XYZ999" in text for text in captured)
