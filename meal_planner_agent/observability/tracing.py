"""OpenTelemetry distributed tracing setup and trace-context correlation helpers."""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from meal_planner_agent.config import AgentSettings, get_settings

_CONFIGURED_PROVIDER: TracerProvider | None = None
_MEMORY_EXPORTER: InMemorySpanExporter | None = None


def configure_tracing(
    settings: AgentSettings | None = None,
    *,
    enable_in_memory_exporter: bool = False,
    extra_exporter: SpanExporter | None = None,
) -> TracerProvider:
    """Configure OpenTelemetry distributed tracing with GCP Cloud Trace / OTLP / InMemory.

    Links parent-child spans from initial user query through coordinator, specialist
    sub-agents, LLM calls, and tool executions.
    """
    global _CONFIGURED_PROVIDER, _MEMORY_EXPORTER

    cfg = settings or get_settings()
    if _CONFIGURED_PROVIDER is None:
        resource = Resource.create(
            {
                "service.name": cfg.otel_service_name,
                "service.version": "1.0.0",
                "deployment.environment": "gcp" if cfg.google_cloud_project else "local",
            }
        )
        provider = TracerProvider(resource=resource)

        if cfg.enable_cloud_trace and cfg.google_cloud_project:
            try:
                from opentelemetry.exporter.cloud_trace import (  # type: ignore[import-untyped]
                    CloudTraceSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(
                        CloudTraceSpanExporter(project_id=cfg.google_cloud_project)
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "Cloud Trace exporter unavailable, continuing with local tracer: %s", exc
                )

        if cfg.otel_exporter_otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-untyped]
                    OTLPSpanExporter,
                )

                provider.add_span_processor(
                    BatchSpanProcessor(
                        OTLPSpanExporter(endpoint=cfg.otel_exporter_otlp_endpoint)
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logging.getLogger(__name__).debug(
                    "OTLP exporter unavailable, continuing with local tracer: %s", exc
                )

        trace.set_tracer_provider(provider)
        _CONFIGURED_PROVIDER = provider

    if enable_in_memory_exporter and _MEMORY_EXPORTER is None:
        _MEMORY_EXPORTER = InMemorySpanExporter()
        _CONFIGURED_PROVIDER.add_span_processor(SimpleSpanProcessor(_MEMORY_EXPORTER))

    if extra_exporter is not None:
        _CONFIGURED_PROVIDER.add_span_processor(SimpleSpanProcessor(extra_exporter))

    return _CONFIGURED_PROVIDER


def get_tracer(instrumentation_name: str = "meal_planner_agent") -> trace.Tracer:
    """Return an OpenTelemetry tracer bound to the configured TracerProvider."""
    configure_tracing()
    return trace.get_tracer(instrumentation_name)


def get_in_memory_span_exporter() -> InMemorySpanExporter:
    """Return (and attach if needed) an InMemorySpanExporter for trace verification."""
    configure_tracing(enable_in_memory_exporter=True)
    assert _MEMORY_EXPORTER is not None
    return _MEMORY_EXPORTER


def get_active_trace_context() -> tuple[str | None, str | None]:
    """Return active hex (trace_id, span_id) for correlating structured logs with traces."""
    span = trace.get_current_span()
    ctx = span.get_span_context() if span else None
    if ctx is None or not ctx.is_valid:
        return None, None
    return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
