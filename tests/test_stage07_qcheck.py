import json

import pytest

from agent.models.question import Question
from agent.pipeline.stage07_qcheck import return_validated_questions


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_return_validated_questions(monkeypatch):
    q_list = [Question(slot_name="dob", text="When were you born?")]

    scored_json = [
        {
            "slot_name": "dob",
            "text": "When were you born?",
            "specificity": 0.8,
            "tacit_power": 0.9,
        }
    ]

    dummy_resp = DummyResponse(content=json.dumps(scored_json))

    async def mock_call(*_args, **_kwargs):
        return dummy_resp

    from agent.llm import openai_client as oc

    monkeypatch.setattr(oc.OpenAIClient, "call", mock_call, raising=True)

    accepted = await return_validated_questions(q_list)
    assert len(accepted) == 1
    assert accepted[0].specificity == 0.8 