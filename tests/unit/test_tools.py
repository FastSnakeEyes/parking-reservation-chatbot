"""Every tool's schema must build the way a real chat model binds it.

The graph tests' fake models skip bind_tools, so an unresolvable annotation
(e.g. ToolRuntime[ChatContext] reaching a TYPE_CHECKING-only import) only
fails against a real model - this catches it without one."""

import pytest
from app.ai.tools import approve_reservation, create_parking_tools
from langchain_core.utils.function_calling import convert_to_openai_tool

TOOLS = [*create_parking_tools(), approve_reservation]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda tool: tool.name)
def test_tool_schema_builds_for_the_model(tool):
    schema = convert_to_openai_tool(tool)

    assert schema["function"]["name"] == tool.name
    # The injected runtime is never exposed to the model as an argument.
    assert "runtime" not in schema["function"]["parameters"].get("properties", {})
