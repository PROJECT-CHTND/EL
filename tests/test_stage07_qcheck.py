import json

import pytest
from prometheus_client import REGISTRY

from agent.models.question import Question
from agent.pipeline.stage07_qcheck import return_validated_questions
from agent.monitoring.metrics import SLOT_DUPLICATE_RATE, SLOT_DUPLICATES_TOTAL


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_return_validated_questions(monkeypatch):
    q_list = [
        Question(slot_name="dob", text="When were you born?"),
        Question(slot_name="dob", text="When were you born?"),
    ]

    scored_json = [
        {
            "slot_name": "dob",
            "text": "When were you born?",
            "specificity": 0.8,
            "tacit_power": 0.9,
        },
        {
            "slot_name": "dob",
            "text": "When were you born?",
            "specificity": 0.9,
            "tacit_power": 0.95,
        },
    ]

    dummy_resp = DummyResponse(content=json.dumps(scored_json))

    async def mock_call(*_args, **_kwargs):
        return dummy_resp

    from agent.llm import openai_client as oc

    monkeypatch.setattr(oc.OpenAIClient, "call", mock_call, raising=True)

    try:
        SLOT_DUPLICATE_RATE.remove("stage07_qcheck")
    except KeyError:
        pass
    try:
        SLOT_DUPLICATES_TOTAL.remove("stage07_qcheck")
    except KeyError:
        pass

    accepted = await return_validated_questions(q_list)
    assert len(accepted) == 1
    assert accepted[0].specificity == 0.8

    rate_value = REGISTRY.get_sample_value("slot_duplicate_rate", {"pipeline_stage": "stage07_qcheck"})
    assert rate_value == pytest.approx(0.5)

    duplicate_counter_after = REGISTRY.get_sample_value(
        "slot_duplicates_total", {"pipeline_stage": "stage07_qcheck"}
    ) or 0.0
    assert duplicate_counter_after == pytest.approx(1.0)