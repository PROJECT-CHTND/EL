import json
import os

import pytest

from agent.models.question import Question
from agent.pipeline.stage07_qcheck import return_validated_questions


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_qcheck_threshold_env_changes(monkeypatch):
    inputs = [
        Question(slot_name="x", text="A", specificity=0.6, tacit_power=0.6),
        Question(slot_name="x", text="B", specificity=0.9, tacit_power=0.9),
    ]
    scored = [
        {"slot_name": "x", "text": "A", "specificity": 0.6, "tacit_power": 0.6},
        {"slot_name": "x", "text": "B", "specificity": 0.9, "tacit_power": 0.9},
    ]
    dummy = DummyResponse(content=json.dumps(scored))

    async def mock_call(*_args, **_kwargs):
        return dummy

    from agent.llm import openai_client as oc

    monkeypatch.setenv("QCHECK_SPEC_THRESHOLD", "0.7")
    monkeypatch.setenv("QCHECK_TACIT_THRESHOLD", "0.7")
    monkeypatch.setattr(oc.OpenAIClient, "call", mock_call, raising=True)
    keep_high = await return_validated_questions(inputs)
    assert len(keep_high) == 1 and keep_high[0].text == "B"

    # lower thresholds → both accepted
    monkeypatch.setenv("QCHECK_SPEC_THRESHOLD", "0.5")
    monkeypatch.setenv("QCHECK_TACIT_THRESHOLD", "0.5")
    keep_both = await return_validated_questions(inputs)
    assert len(keep_both) == 2


