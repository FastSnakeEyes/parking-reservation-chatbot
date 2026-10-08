from enum import StrEnum


class ChatStatus(StrEnum):
    COMPLETE = "complete"
    # The conversation is paused for the administrator's decision on its reservation.
    WAITING_FOR_ADMIN = "waiting_for_admin"
