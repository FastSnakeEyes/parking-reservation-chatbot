from enum import StrEnum


class ReservationStatus(StrEnum):
    PENDING = "pending"  # submitted, waiting for the administrator
    APPROVED = "approved"  # approved by the administrator and written through the MCP server
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class ApprovalDecision(StrEnum):
    """The administrator's decision on a pending reservation."""

    APPROVED = "approved"
    REJECTED = "rejected"
