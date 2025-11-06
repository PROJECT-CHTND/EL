"""Monitoring utilities: Prometheus instrumentation and custom metrics."""

from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore

from agent.monitoring.metrics import (  # noqa: F401 - re-exported for convenience
    SLOT_COVERAGE,
    SLOT_DUPLICATE_RATE,
    SLOT_DUPLICATES_TOTAL,
    SLOT_GAP_LATENCY_SECONDS,
)


def attach_instrumentator(app):  # noqa: D401
    """Attach Prometheus Instrumentator to FastAPI app and expose /metrics."""

    instrumentator = Instrumentator()
    instrumentator.instrument(app)
    instrumentator.expose(app)
    return instrumentator


__all__ = [
    "attach_instrumentator",
    "SLOT_COVERAGE",
    "SLOT_DUPLICATE_RATE",
    "SLOT_DUPLICATES_TOTAL",
    "SLOT_GAP_LATENCY_SECONDS",
]
