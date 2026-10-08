from enum import StrEnum


class ReservationStatus(StrEnum):
    PENDING = "pending"  # submitted, waiting for the administrator
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    RECORDED = "recorded"  # approved and written through the MCP server


class ApprovalDecision(StrEnum):
    """The administrator's decision on a pending reservation."""

    APPROVED = "approved"
    REJECTED = "rejected"
