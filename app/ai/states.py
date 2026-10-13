"""LangGraph / agent state schemas.

- ParkingState: ParkingGraph's state (app/ai/graph.py), one checkpointer
  thread per chat session.
- UserAgentState: the user agent's state (app/ai/agents.py), run as
  ParkingGraph's `user_agent` node.

The admin agent uses LangChain's default AgentState. The per-request
dependencies passed alongside the state are ChatContext (app/ai/tools.py),
the graph's and agents' runtime context, not state.
"""

from typing import Annotated, NotRequired, TypedDict

from langchain.agents import AgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# Enum fields hold the enum's plain string value: the checkpointer only loads
# its own registered types back, so enum members must not be persisted.
class ParkingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: NotRequired[dict | None]  # set by prepare_reservation, consumed by `confirm`
    last_reservation_id: NotRequired[str | None]
    decision: NotRequired[str | None]  # an ApprovalDecision value, set by ParkingGraph.resume
    admin_note: NotRequired[str | None]
    reservation_status: NotRequired[str | None]  # a ReservationStatus value
    notice_pending: NotRequired[bool]  # the decision notice is still undelivered
    submitted: NotRequired[bool]  # this turn's `confirm` saved a booking: route to `admin`


class UserAgentState(AgentState):
    draft: NotRequired[dict | None]  # set by prepare_reservation's Command
