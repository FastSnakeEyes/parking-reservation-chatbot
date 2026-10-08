from enum import StrEnum


class PresidioEntity(StrEnum):
    """Presidio analyzer entity types used by PIIGuard (app/ai/guardrails.py)."""

    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    IBAN_CODE = "IBAN_CODE"
    US_SSN = "US_SSN"
    IP_ADDRESS = "IP_ADDRESS"
    PERSON = "PERSON"


class PIIType(StrEnum):
    """PII types redacted by the user agent's PIIMiddleware stack
    (app/ai/guardrails.py's pii_middleware). EMAIL and CREDIT_CARD use
    LangChain's built-in detectors, which are looked up by these names."""

    EMAIL = "email"
    CREDIT_CARD = "credit_card"
    PHONE = "phone"
    NAME = "name"
    SURNAME = "surname"
    CAR_NUMBER = "car_number"
