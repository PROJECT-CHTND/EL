import os
import socket
import time

from agent.monitoring.metrics import start_metrics_server, set_slot_coverage, observe_turn_latency
from prometheus_client import generate_latest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_metrics_server_idempotent_and_metrics_recorded(monkeypatch):
    port = _free_port()
    monkeypatch.setenv("METRICS_PORT", str(port))

    # should not raise even if called twice
    start_metrics_server()
    start_metrics_server()

    # record a couple of metrics (does not require HTTP fetch)
    set_slot_coverage(0.75)
    observe_turn_latency(0.42)

    # registry should include our metrics names
    output = generate_latest()
    assert b"slot_coverage" in output
    assert b"turn_latency_seconds" in output


