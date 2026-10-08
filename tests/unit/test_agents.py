import pytest
from app.ai.agents import parse_admin_reply
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class FakeStructuredOutputModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel has no bind_tools - with_structured_output's
    default (function-calling) strategy needs one, so a fake model used
    with it needs a no-op override."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _decision_reply(**fields) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "AdminDecision", "args": fields, "id": "call_1"}])


@pytest.mark.asyncio
async def test_parse_admin_reply_approved():
    llm = FakeStructuredOutputModel(responses=[_decision_reply(approved=True, reason="")])

    decision = parse_admin_reply(llm, "ok, approved")

    assert decision == {"approved": True, "reason": ""}


@pytest.mark.asyncio
async def test_parse_admin_reply_rejected_with_reason():
    llm = FakeStructuredOutputModel(responses=[_decision_reply(approved=False, reason="lot is closed")])

    decision = parse_admin_reply(llm, "no, the lot is closed today")

    assert decision == {"approved": False, "reason": "lot is closed"}


def test_public_contacts_are_not_redacted(monkeypatch):
    from app.ai.agents import _detect_phone, _skip_public_contacts, detect_email
    from app.config import settings

    monkeypatch.setattr(settings, "public_contacts", ["+370 5 200 0000", "info@vilpark.lt"])
    text = "Support: +370 5 200 0000, info@vilpark.lt. Customer: +370 612 34567, jonas@gmail.com"

    phones = [m["value"] for m in _skip_public_contacts(_detect_phone)(text)]
    emails = [m["value"] for m in _skip_public_contacts(detect_email)(text)]

    assert phones == ["+370 612 34567"]
    assert emails == ["jonas@gmail.com"]
