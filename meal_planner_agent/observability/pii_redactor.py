"""Active PII scrubbing pipeline with Google Cloud DLP integration and local fallback."""

from __future__ import annotations

import logging
import re
from typing import Any

from meal_planner_agent.config import AgentSettings, get_settings

# Ordered regex scrubbers for sensitive personal and credential data
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("API_KEY", re.compile(r"\b(?:AIza[0-9A-Za-z_-]{30,45}|sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (
        "PHONE",
        re.compile(r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\b\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"),
    ),
    (
        "STREET_ADDRESS",
        re.compile(
            r"\b\d{1,5}\s+[A-Za-z0-9.\s]{2,30}\s+"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Court|Ct|Way|Parkway|Pkwy)\b\.?",
            re.IGNORECASE,
        ),
    ),
]

_SENSITIVE_FIELD_NAMES = {
    "api_key",
    "token",
    "password",
    "secret",
    "credit_card",
    "card_number",
    "ssn",
    "phone",
    "phone_number",
    "email",
    "street_address",
    "home_address",
}


class PIIRedactor:
    """Scrubs sensitive PII from text and nested payloads prior to logging or persistence.

    When ``enable_cloud_dlp_redaction=True`` and a GCP project is configured, attempts
    de-identification via Google Cloud Sensitive Data Protection (Cloud DLP API) and
    always applies deterministic local regex redaction as defense-in-depth.
    """

    def __init__(
        self,
        settings: AgentSettings | None = None,
        dlp_client: Any | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._dlp_client = dlp_client

    def _get_dlp_client(self) -> Any | None:
        if self._dlp_client is not None:
            return self._dlp_client
        if not (self._settings.enable_cloud_dlp_redaction and self._settings.google_cloud_project):
            return None
        try:
            from google.cloud import dlp_v2  # type: ignore[import-untyped]

            self._dlp_client = dlp_v2.DlpServiceClient()
            return self._dlp_client
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug(
                "Cloud DLP client unavailable, using local regex PII redactor: %s", exc
            )
            return None

    def _redact_with_cloud_dlp(self, text: str) -> str:
        client = self._get_dlp_client()
        if client is None or not text.strip():
            return text
        try:
            parent = f"projects/{self._settings.google_cloud_project}"
            inspect_config = {
                "info_types": [
                    {"name": "EMAIL_ADDRESS"},
                    {"name": "PHONE_NUMBER"},
                    {"name": "CREDIT_CARD_NUMBER"},
                    {"name": "US_SOCIAL_SECURITY_NUMBER"},
                    {"name": "STREET_ADDRESS"},
                ]
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {
                                "replace_with_info_type_config": {}
                            }
                        }
                    ]
                }
            }
            response = client.deidentify_content(
                request={
                    "parent": parent,
                    "deidentify_config": deidentify_config,
                    "inspect_config": inspect_config,
                    "item": {"value": text},
                }
            )
            return str(response.item.value)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug(
                "Cloud DLP deidentify_content failed, applying local redaction: %s", exc
            )
            return text

    def redact_text(self, text: str) -> str:
        """Scrub all recognized PII and credential patterns from a string."""
        if not text:
            return text
        scrubbed = self._redact_with_cloud_dlp(text)
        for label, pattern in _PII_PATTERNS:
            scrubbed = pattern.sub(f"[REDACTED_{label}]", scrubbed)
        return scrubbed

    def redact_payload(self, value: Any) -> Any:
        """Recursively redact strings and sensitive dictionary keys in structured payloads."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                if key_str.lower() in _SENSITIVE_FIELD_NAMES:
                    cleaned[key_str] = "[REDACTED_SENSITIVE_FIELD]"
                else:
                    cleaned[key_str] = self.redact_payload(item)
            return cleaned
        if isinstance(value, list):
            return [self.redact_payload(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_payload(item) for item in value)
        return value


_DEFAULT_REDACTOR: PIIRedactor | None = None


def get_pii_redactor() -> PIIRedactor:
    """Return a shared PIIRedactor instance."""
    global _DEFAULT_REDACTOR
    if _DEFAULT_REDACTOR is None:
        _DEFAULT_REDACTOR = PIIRedactor()
    return _DEFAULT_REDACTOR
