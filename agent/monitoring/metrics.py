"""Prometheus metrics definitions and helpers for the agent pipeline."""

import os
import threading
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_PIPELINE_STAGE_LABEL = ("pipeline_stage",)
_STARTED = False
_LOCK = threading.Lock()

# Gauge: ratio of filled slots in the registry per pipeline stage
SLOT_COVERAGE = Gauge(
    "slot_coverage",
    "Proportion of registered slots considered filled.",
    labelnames=_PIPELINE_STAGE_LABEL,
)

# Gauge: ratio of duplicate questions filtered out per pipeline stage
SLOT_DUPLICATE_RATE = Gauge(
    "slot_duplicate_rate",
    "Ratio of duplicate or filtered questions removed during validation.",
    labelnames=_PIPELINE_STAGE_LABEL,
)

# Counter: number of duplicate questions filtered out
SLOT_DUPLICATES_TOTAL = Counter(
    "slot_duplicates_total",
    "Total count of duplicate questions removed during validation.",
    labelnames=_PIPELINE_STAGE_LABEL,
)

# Gauge: latency measurement for slot gap prioritisation per stage
SLOT_GAP_LATENCY_SECONDS = Gauge(
    "slot_gap_latency_seconds",
    "Processing latency in seconds for slot gap prioritisation logic.",
    labelnames=_PIPELINE_STAGE_LABEL,
)

# Histogram: one-turn latency (for Discord flow)
TURN_LATENCY_SECONDS = Histogram(
    "turn_latency_seconds",
    "End-to-end latency per turn (seconds).",
    buckets=(0.2, 0.5, 1, 2, 3, 5, 10, 20),
)


# ---- Helper functions (backward-compatible with earlier code) ----
def start_metrics_server(port: Optional[int] = None) -> None:
    """Start Prometheus metrics HTTP server if not started."""
    global _STARTED
    if _STARTED:
        return
    p = int(os.getenv("METRICS_PORT", str(port or 8000)))
    with _LOCK:
        if not _STARTED:
            start_http_server(p)
            _STARTED = True


def set_slot_coverage(value: float, *, pipeline_stage: str = "bot") -> None:
    """Set slot coverage gauge for a given stage label."""
    SLOT_COVERAGE.labels(pipeline_stage=pipeline_stage).set(max(0.0, min(1.0, value)))


def observe_turn_latency(seconds: float) -> None:
    """Record one-turn latency."""
    TURN_LATENCY_SECONDS.observe(max(0.0, seconds))

