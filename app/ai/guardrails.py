"""Guardrails: everything PII and prompt-injection related lives here.

- PIIGuard (Presidio + spaCy): blocks personal data in vector-search queries
  and redacts it from static docs at ingestion.
- pii_middleware(): the user agent's PIIMiddleware stack (app/ai/agents.py),
  redacting PII from the customer's input, tool results and answers.
- is_prompt_injection(): opt-in via settings.use_injection_model, since the
  classifier model requires a torch/tensorflow backend that isn't installed
  by default.

The operator's own published contacts (settings.public_contacts) are never
treated as PII by any of them.
"""

import re
from collections.abc import Callable
from functools import lru_cache

from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware._redaction import PIIMatch, detect_email
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PII_ENTITIES = [
    # No PERSON: en_core_web_sm's NER is unreliable on this domain's many
    # Lithuanian place names (observed: "Vilnius", "Email", "Snipiskes" all
    # misclassified as PERSON), and there's no real PERSON-leak risk to
    # catch here - the vector store holds only public static docs, and
    # reservation PII never flows through the redacted answer/query paths.
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "IP_ADDRESS",
]


def _normalize_contact(value: str) -> str:
    return re.sub(r"[\s-]", "", value).lower()


def is_public_contact(value: str) -> bool:
    """The operator's own published phone/email (settings.public_contacts, e.g.
    the support line) is meant to be shared, not redacted."""
    return _normalize_contact(value) in {_normalize_contact(c) for c in settings.public_contacts}


class PIIGuard:
    def __init__(self):
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": settings.spacy_model}],
            }
        )
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"],
        )
        self._anonymizer = AnonymizerEngine()

    @property
    def analyzer(self) -> AnalyzerEngine:
        """Exposed so other PII detectors (e.g. pii_middleware's PERSON
        detection below) reuse this already-configured engine instead of building
        a second one with Presidio's default (much larger, not-installed)
        spaCy model."""
        return self._analyzer

    def analyze(
        self,
        text: str,
        entities: list[str] | None = None,
    ) -> list[RecognizerResult]:
        if not text:
            return []

        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=entities or DEFAULT_PII_ENTITIES,
        )

        return [result for result in results if not is_public_contact(text[result.start : result.end])]

    def contains_pii(
        self,
        text: str,
        entities: list[str] | None = None,
    ) -> bool:
        return bool(self.analyze(text, entities))

    def redact(
        self,
        text: str,
        entities: list[str] | None = None,
    ) -> str:
        if not text:
            return text

        results = self.analyze(text, entities)

        if not results:
            return text

        # Entity types only - never log the matched PII values themselves.
        logger.info("Redacted PII from text: %s", sorted({result.entity_type for result in results}))

        return self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        ).text


@lru_cache
def get_pii_guard() -> PIIGuard:
    return PIIGuard()


PHONE_PATTERN = r"(?:\+\d[\d -]{7,14}\d|\b8[ -]?6\d{2}[ -]?\d{5}\b)"  # +370 612 34567, 8 612 34567
CAR_NUMBER_PATTERN = r"\b(?!EUR|USD)[A-Z]{3}[ -]?\d{3}\b"  # ABC123, ABC 123, ABC-123


def _skip_public_contacts(detector: Callable[[str], list[PIIMatch]]) -> Callable[[str], list[PIIMatch]]:
    def detect(content: str) -> list[PIIMatch]:
        return [m for m in detector(content) if not is_public_contact(m["value"])]

    return detect


@lru_cache(maxsize=256)
def _person_parts(content: str) -> tuple[tuple[str, int, int], ...]:
    """(part, start, end) for each detected person: first word = name, the rest = surname.
    Cached so Presidio runs once per text, not once per detector."""
    parts = []
    for r in get_pii_guard().analyzer.analyze(text=content, entities=["PERSON"], language="en", score_threshold=0.6):
        words = list(re.finditer(r"\S+", content[r.start : r.end]))
        if not words:
            continue
        parts.append(("name", r.start + words[0].start(), r.start + words[0].end()))
        if len(words) > 1:
            parts.append(("surname", r.start + words[1].start(), r.start + words[-1].end()))
    return tuple(parts)


def _detect_phone(content: str) -> list[PIIMatch]:
    return [
        {"type": "phone", "value": m.group(), "start": m.start(), "end": m.end()}
        for m in re.finditer(PHONE_PATTERN, content)
    ]


def _detect_name(content: str) -> list[dict]:
    return [
        {"type": "name", "value": content[s:e], "start": s, "end": e}
        for part, s, e in _person_parts(content)
        if part == "name"
    ]


def _detect_surname(content: str) -> list[dict]:
    return [
        {"type": "surname", "value": content[s:e], "start": s, "end": e}
        for part, s, e in _person_parts(content)
        if part == "surname"
    ]


def pii_middleware() -> list[PIIMiddleware]:
    """Redact personal data before it reaches the model or the customer.

    Contact and payment data (email, phone, card) is never needed for a reservation,
    so it is redacted everywhere: customer input, tool results and the model's answer -
    except the operator's own settings.public_contacts, which customers may ask for.

    Name, surname and car number ARE needed for the customer's own reservation, so they
    are redacted only in tool results. That hides other customers' data coming back from
    Milvus / PostgreSQL while the customer's own details still pass through."""
    everywhere = {"strategy": "redact", "apply_to_output": True, "apply_to_tool_results": True}
    tool_results_only = {
        "strategy": "redact",
        "apply_to_input": False,
        "apply_to_output": False,
        "apply_to_tool_results": True,
    }

    return [
        PIIMiddleware("email", detector=_skip_public_contacts(detect_email), **everywhere),
        PIIMiddleware("credit_card", **everywhere),
        PIIMiddleware("phone", detector=_skip_public_contacts(_detect_phone), **everywhere),
        PIIMiddleware("name", detector=_detect_name, **tool_results_only),
        PIIMiddleware("surname", detector=_detect_surname, **tool_results_only),
        PIIMiddleware("car_number", detector=CAR_NUMBER_PATTERN, **tool_results_only),
    ]


@lru_cache
def get_injection_classifier():
    # Imported lazily: requires a torch/tensorflow backend not installed by
    # default, since this guard is opt-in (settings.use_injection_model).
    from transformers import pipeline

    return pipeline("text-classification", model=settings.injection_model)


def is_prompt_injection(text: str) -> bool:
    if not settings.use_injection_model or not text:
        return False

    result = get_injection_classifier()(text, truncation=True)[0]
    flagged = "INJECT" in result["label"].upper() and result["score"] >= 0.5

    if flagged:
        logger.warning("Injection classifier flagged input (score=%.2f)", result["score"])

    return flagged
