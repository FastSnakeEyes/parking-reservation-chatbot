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
from app.constants import (
    CAR_NUMBER_PATTERN,
    INJECTION_LABEL,
    INJECTION_SCORE_THRESHOLD,
    PERSON_PARTS_CACHE_SIZE,
    PERSON_SCORE_THRESHOLD,
    PHONE_PATTERN,
    PII_ENTITIES,
    PII_LANGUAGE,
    PII_REDACT_EVERYWHERE,
    PII_REDACT_TOOL_RESULTS_ONLY,
)
from app.enums import PIIType, PresidioEntity
from app.logger import get_logger

logger = get_logger(__name__)


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
                "models": [{"lang_code": PII_LANGUAGE, "model_name": settings.spacy_model}],
            }
        )
        self._analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=[PII_LANGUAGE],
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
            language=PII_LANGUAGE,
            entities=entities or list(PII_ENTITIES),
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


def _skip_public_contacts(detector: Callable[[str], list[PIIMatch]]) -> Callable[[str], list[PIIMatch]]:
    def detect(content: str) -> list[PIIMatch]:
        return [m for m in detector(content) if not is_public_contact(m["value"])]

    return detect


@lru_cache(maxsize=PERSON_PARTS_CACHE_SIZE)
def _person_parts(content: str) -> tuple[tuple[str, int, int], ...]:
    """(part, start, end) for each detected person: first word = name, the rest = surname.
    Cached so Presidio runs once per text, not once per detector."""
    parts = []
    results = get_pii_guard().analyzer.analyze(
        text=content,
        entities=[PresidioEntity.PERSON],
        language=PII_LANGUAGE,
        score_threshold=PERSON_SCORE_THRESHOLD,
    )
    for r in results:
        words = list(re.finditer(r"\S+", content[r.start : r.end]))
        if not words:
            continue
        parts.append((PIIType.NAME, r.start + words[0].start(), r.start + words[0].end()))
        if len(words) > 1:
            parts.append((PIIType.SURNAME, r.start + words[1].start(), r.start + words[-1].end()))
    return tuple(parts)


def _detect_phone(content: str) -> list[PIIMatch]:
    return [
        {"type": PIIType.PHONE, "value": m.group(), "start": m.start(), "end": m.end()}
        for m in re.finditer(PHONE_PATTERN, content)
    ]


def _detect_name(content: str) -> list[dict]:
    return [
        {"type": PIIType.NAME, "value": content[s:e], "start": s, "end": e}
        for part, s, e in _person_parts(content)
        if part == PIIType.NAME
    ]


def _detect_surname(content: str) -> list[dict]:
    return [
        {"type": PIIType.SURNAME, "value": content[s:e], "start": s, "end": e}
        for part, s, e in _person_parts(content)
        if part == PIIType.SURNAME
    ]


def pii_middleware() -> list[PIIMiddleware]:
    """Redact personal data before it reaches the model or the customer.

    Contact and payment data (email, phone, card) is never needed for a reservation,
    so it is redacted everywhere: customer input, tool results and the model's answer -
    except the operator's own settings.public_contacts, which customers may ask for.

    Name, surname and car number ARE needed for the customer's own reservation, so they
    are redacted only in tool results. That hides other customers' data coming back from
    Milvus / PostgreSQL while the customer's own details still pass through."""
    return [
        PIIMiddleware(PIIType.EMAIL, detector=_skip_public_contacts(detect_email), **PII_REDACT_EVERYWHERE),
        PIIMiddleware(PIIType.CREDIT_CARD, **PII_REDACT_EVERYWHERE),
        PIIMiddleware(PIIType.PHONE, detector=_skip_public_contacts(_detect_phone), **PII_REDACT_EVERYWHERE),
        PIIMiddleware(PIIType.NAME, detector=_detect_name, **PII_REDACT_TOOL_RESULTS_ONLY),
        PIIMiddleware(PIIType.SURNAME, detector=_detect_surname, **PII_REDACT_TOOL_RESULTS_ONLY),
        PIIMiddleware(PIIType.CAR_NUMBER, detector=CAR_NUMBER_PATTERN, **PII_REDACT_TOOL_RESULTS_ONLY),
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
    flagged = INJECTION_LABEL in result["label"].upper() and result["score"] >= INJECTION_SCORE_THRESHOLD

    if flagged:
        logger.warning("Injection classifier flagged input (score=%.2f)", result["score"])

    return flagged
