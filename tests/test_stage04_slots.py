import json

import pytest

from agent.models.kg import Entity, Relation, KGPayload
from agent.pipeline.stage04_slots import propose_slots
from agent.slots import SlotRegistry


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
    slots = await propose_slots(kg, registry=registry)

    assert len(slots) == 2
    assert registry.unfilled_slots()[0].name == "date_of_birth" 