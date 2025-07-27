import json

import pytest

from agent.slots import Slot
from agent.pipeline.stage06_qgen import generate_questions


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_generate_questions(monkeypatch, tmp_path):
    slots = [Slot(name="date_of_birth", description="Person's birth date", type="attribute")]

    questions_json = [
        {"slot_name": "date_of_birth", "text": "When were you born?"}
    ]
    dummy_resp = DummyResponse(content=json.dumps(questions_json))

    async def mock_call(*_args, **_kwargs):
        return dummy_resp

    from agent.llm import openai_client as oc

    monkeypatch.setattr(oc.OpenAIClient, "call", mock_call, raising=True)

    q_list = await generate_questions(slots)

    assert len(q_list) == 1
    assert q_list[0].text.startswith("When") 