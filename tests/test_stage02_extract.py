import json
from types import SimpleNamespace

import pytest

from agent.pipeline.stage02_extract import extract_knowledge


class DummyFunctionCall:
    def __init__(self, arguments: str):
        self.arguments = arguments


class DummyResponse:
    def __init__(self, function_call=None, content=""):
        self.function_call = function_call
        self.content = content


@pytest.mark.asyncio
async def test_extract_with_function_call(monkeypatch):
    """Ensure extract_knowledge parses function_call JSON correctly."""

    payload_json = {"entities": [{"id": "e1", "label": "Test"}], "relations": []}
    dummy_resp = DummyResponse(function_call=DummyFunctionCall(json.dumps(payload_json)))

    async def mock_call(*_args, **_kwargs):  # noqa: D401
        return dummy_resp

    # Patch OpenAIClient.call
    from agent import llm as llm_pkg  # dynamic import to access module
    monkeypatch.setattr(
        llm_pkg.openai_client.OpenAIClient, "call", mock_call, raising=True
    )

    result = await extract_knowledge("dummy text")
    assert result.entities[0].label == "Test"
    assert result.relations == [] 