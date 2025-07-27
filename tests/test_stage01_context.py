import json
from unittest.mock import AsyncMock

import pytest

from agent.pipeline.stage01_context import generate_context


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_generate_context(monkeypatch):
    """Context generation should parse JSON and not fail."""

    dummy_json = {
        "rag_keys": "key1;key2",
        "mid_summary": "mid",
        "global_summary": "global",
    }
    dummy_resp = DummyResponse(content=json.dumps(dummy_json))

    async def mock_call(*_args, **_kwargs):
        return dummy_resp

    # Patch OpenAIClient.call
    from agent.llm import openai_client as oc

    monkeypatch.setattr(oc.OpenAIClient, "call", mock_call, raising=True)

    # Patch redis to noop
    async def mock_xadd(*_args, **_kwargs):
        return None

    async def mock_aclose():
        return None

    class DummyRedis:
        xadd = staticmethod(mock_xadd)
        aclose = staticmethod(mock_aclose)

    monkeypatch.setattr("agent.pipeline.stage01_context.aioredis.from_url", lambda *_args, **_kwargs: DummyRedis())

    payload = await generate_context("some text")
    assert payload.rag_keys == "key1;key2"
    assert payload.mid_summary == "mid"
    assert payload.global_summary == "global" 