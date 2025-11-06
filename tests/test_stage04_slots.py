import json

import pytest

from prometheus_client import REGISTRY

from agent.models.kg import Entity, Relation, KGPayload
from agent.pipeline.stage04_slots import propose_slots
from agent.slots import SlotRegistry
from agent.monitoring.metrics import SLOT_COVERAGE


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_propose_slots(monkeypatch):
    kg = KGPayload(entities=[Entity(id="e1", label="A")], relations=[Relation(source="e1", target="e1", type="SELF")])

    suggestions = [
        {"name": "date_of_birth", "description": "Add date of birth info"},
        {"name": "nationality", "description": "Add nationality detail"},
    ]

    dummy_resp = DummyResponse(content=json.dumps(suggestions))

    async def mock_call(*_args, **_kwargs):
        return dummy_resp

    from agent.llm import openai_client as oc

    monkeypatch.setattr(oc.OpenAIClient, "call", mock_call, raising=True)

    registry = SlotRegistry()
    # Clear any existing metric sample for deterministic assertions
    try:
        SLOT_COVERAGE.remove("stage04_slots")
    except KeyError:
        pass
    slots = await propose_slots(kg, registry=registry)

    assert len(slots) == 2
    assert registry.unfilled_slots()[0].name == "date_of_birth"

    coverage_value = REGISTRY.get_sample_value("slot_coverage", {"pipeline_stage": "stage04_slots"})
    assert coverage_value == pytest.approx(registry.coverage())