"""Prometheus metrics definitions for the agent pipeline."""

from prometheus_client import Counter, Gauge

_PIPELINE_STAGE_LABEL = ("pipeline_stage",)

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

