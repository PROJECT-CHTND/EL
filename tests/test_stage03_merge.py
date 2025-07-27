import pytest

from agent.models.kg import Entity, Relation, KGPayload
from agent.pipeline.stage03_merge import calculate_confidence, merge_and_persist


def test_calculate_confidence():
    conf = calculate_confidence([-1.0, -0.5], stage_weight=0.8)
    assert 0.0 <= conf <= 0.8


def test_merge_and_persist(monkeypatch):
    entities = [Entity(id="e1", label="Alpha")]
    relations = [Relation(source="e1", target="e1", type="SELF")]
    payload = KGPayload(entities=entities, relations=relations)

    calls = {"count": 0}

    class DummyClient:
        def ingest(self, p):
            calls["payload"] = p
            calls["count"] += 1

    monkeypatch.setattr("agent.pipeline.stage03_merge.Neo4jClient", lambda: DummyClient())

    result = merge_and_persist(payload, logprobs=[-1.0])
    
    # Verify ingest was called
    assert calls["count"] == 1
    assert calls["payload"] is result
    assert result.entities[0].confidence is not None
    # Verify confidence is within expected range
    assert 0.0 <= result.entities[0].confidence <= 1.0