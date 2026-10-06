"""Guardrails: block personal data leaking into vector search, and screen
prompt injection in user input.

PIIGuard runs on every request (Presidio + spaCy). InjectionGuard is opt-in
via settings.use_injection_model, since the classifier model requires a
torch/tensorflow backend that isn't installed by default.
"""

from functools import lru_cache

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

        # The operator's own published phone/email (e.g. the support line)
        # is meant to be shared, not redacted.
        public_contacts = {contact.strip() for contact in settings.public_contacts}

        if not public_contacts:
            return results

        return [result for result in results if text[result.start : result.end].strip() not in public_contacts]

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
