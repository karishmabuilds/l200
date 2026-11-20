"""Structured JSON logging with automatic PII scrubbing and OpenTelemetry correlation."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from meal_planner_agent.config import get_settings
from meal_planner_agent.observability.pii_redactor import PIIRedactor, get_pii_redactor
from meal_planner_agent.observability.tracing import get_active_trace_context


class StructuredJsonFormatter(logging.Formatter):
    """Formats every log record as Cloud Logging-compatible, PII-scrubbed JSON."""

    def __init__(
        self,
        service_name: str = "meal-planning-concierge-agent",
        redactor: PIIRedactor | None = None,
    ) -> None:
        super().__init__()
        self._service_name = service_name
        self._redactor = redactor or get_pii_redactor()

    def format(self, record: logging.LogRecord) -> str:
        trace_id, span_id = get_active_trace_context()
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "service": self._service_name,
            "logger": record.name,
            "message": self._redactor.redact_text(record.getMessage()),
        }
        if trace_id:
            payload["trace_id"] = trace_id
        if span_id:
            payload["span_id"] = span_id

        extra_fields = getattr(record, "structured_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(self._redactor.redact_payload(extra_fields))

        if record.exc_info:
            payload["exception"] = self._redactor.redact_text(
                self.formatException(record.exc_info)
            )
        return json.dumps(payload, default=str, ensure_ascii=False)


def get_structured_logger(
    name: str = "meal_planner_agent",
    *,
    stream: Any | None = None,
    redactor: PIIRedactor | None = None,
) -> logging.Logger:
    """Return a logger configured exclusively with StructuredJsonFormatter."""
    settings = get_settings()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(
            StructuredJsonFormatter(
                service_name=settings.otel_service_name,
                redactor=redactor,
            )
        )
        logger.addHandler(handler)
    return logger


def log_structured_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    *,
    level: int = logging.INFO,
    **metadata: Any,
) -> None:
    """Emit a rich structured JSON log record with PII scrubbing and trace correlation."""
    logger.log(
        level,
        message,
        extra={"structured_fields": {"event_type": event_type, **metadata}},
    )
