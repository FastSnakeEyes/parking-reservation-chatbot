"""PII and prompt-injection guardrails (app/ai/guardrails.py)."""

from app.enums import PresidioEntity

PII_LANGUAGE = "en"

PII_ENTITIES = (
    # No PERSON: en_core_web_sm's NER is unreliable on this domain's many
    # Lithuanian place names (observed: "Vilnius", "Email", "Snipiskes" all
    # misclassified as PERSON), and there's no real PERSON-leak risk to
    # catch here - the vector store holds only public static docs, and
    # reservation PII never flows through the redacted answer/query paths.
    PresidioEntity.EMAIL_ADDRESS,
    PresidioEntity.PHONE_NUMBER,
    PresidioEntity.CREDIT_CARD,
    PresidioEntity.IBAN_CODE,
    PresidioEntity.US_SSN,
    PresidioEntity.IP_ADDRESS,
)

# Minimum Presidio confidence for a PERSON match to be redacted as a name/surname.
PERSON_SCORE_THRESHOLD = 0.6
# Texts whose PERSON matches are cached, so Presidio runs once per text, not per detector.
PERSON_PARTS_CACHE_SIZE = 256

PHONE_PATTERN = r"(?:\+\d[\d -]{7,14}\d|\b8[ -]?6\d{2}[ -]?\d{5}\b)"  # +370 612 34567, 8 612 34567
CAR_NUMBER_PATTERN = r"\b(?!EUR|USD)[A-Z]{3}[ -]?\d{3}\b"  # ABC123, ABC 123, ABC-123

# PIIMiddleware options: contact/payment data is redacted everywhere, while the
# customer's own name/surname/car number is redacted only in tool results.
PII_REDACT_EVERYWHERE = {
    "strategy": "redact",
    "apply_to_output": True,
    "apply_to_tool_results": True,
}
PII_REDACT_TOOL_RESULTS_ONLY = {
    "strategy": "redact",
    "apply_to_input": False,
    "apply_to_output": False,
    "apply_to_tool_results": True,
}

# Prompt-injection classifier
INJECTION_LABEL = "INJECT"
INJECTION_SCORE_THRESHOLD = 0.5
