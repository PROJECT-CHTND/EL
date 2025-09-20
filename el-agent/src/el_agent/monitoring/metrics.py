from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


# Histogram: request latency per endpoint
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency per endpoint",
    labelnames=("endpoint",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Counter: LLM calls labeled by kind
LLM_CALLS = Counter("llm_calls_total", "LLM calls", labelnames=("kind",))

# Counter: retrieval calls labeled by pipeline stage
RETRIEVAL_CALLS = Counter("retrieval_calls_total", "Retrieval calls", labelnames=("stage",))

# Gauge: number of open hypotheses
HYPOTHESES_OPEN = Gauge("hypotheses_open", "Number of open hypotheses")


