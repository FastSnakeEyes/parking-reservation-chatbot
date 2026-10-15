"""Every tool's schema must build the way a real chat model binds it.

The graph tests' fake models skip bind_tools, so an unresolvable annotation
(e.g. ToolRuntime[ChatContext] reaching a TYPE_CHECKING-only import) only
fails against a real model - this catches it without one."""

import pytest
from app.ai.tools import create_parking_tools, request_admin_approval
from langchain_core.utils.function_calling import convert_to_openai_tool

TOOLS = [*create_parking_tools(), request_admin_approval]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_tool_schema_builds_for_the_model(tool):
    schema = convert_to_openai_tool(tool)

    assert schema["function"]["name"] == tool.name
    # The injected runtime is never exposed to the model as an argument.
    assert "runtime" not in schema["function"]["parameters"].get("properties", {})


def test_review_description_falls_back_without_a_summary():
    from app.ai.agents import _review_description

    with_summary = {"name": "request_admin_approval", "args": {"summary": " Approve Jane? "}, "id": "1"}
    without = {"name": "request_admin_approval", "args": {}, "id": "2"}

    assert _review_description(with_summary, None, None) == "Approve Jane?"
    assert _review_description(without, None, None) == "Please review this reservation request."


@pytest.mark.parametrize(
    ("query", "cleaned"),
    [
        ("cancellation policy VilniusPark", "cancellation policy"),
        ("VilniusPark minimum reservation length", "minimum reservation length"),
        ("Vilnius Park rules", "rules"),
        ("vilniuspark", "vilniuspark"),  # nothing else to search for: kept
        ("parking in Vilnius old town", "parking in Vilnius old town"),  # the city stays
    ],
)
def test_search_queries_drop_the_brand_name(query, cleaned):
    """With the brand name, queries matched every doc's generic intro chunk
    instead of the relevant one (found by the Evidently evaluation)."""
    from app.ai.tools import clean_search_query

    assert clean_search_query(query) == cleaned
