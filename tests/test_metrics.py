from prometheus_client import generate_latest

from agent.monitoring.metrics import (
    SLOT_COVERAGE,
    SLOT_DUPLICATE_RATE,
    SLOT_GAP_LATENCY_SECONDS,
)


def test_metrics_include_pipeline_stage_labels():
    SLOT_COVERAGE.labels(pipeline_stage="metrics_test").set(0.42)
    SLOT_DUPLICATE_RATE.labels(pipeline_stage="metrics_test").set(0.25)
    SLOT_GAP_LATENCY_SECONDS.labels(pipeline_stage="metrics_test").set(0.1)

    output = generate_latest()

    assert b'slot_coverage{pipeline_stage="metrics_test"}' in output
    assert b'slot_duplicate_rate{pipeline_stage="metrics_test"}' in output
    assert b'slot_gap_latency_seconds{pipeline_stage="metrics_test"}' in output
