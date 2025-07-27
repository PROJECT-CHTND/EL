"""Monitoring utilities: Prometheus instrumentation."""

from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore


def attach_instrumentator(app):  # noqa: D401
    """Attach Prometheus Instrumentator to FastAPI app."""
    Instrumentator().instrument(app).expose(app) 