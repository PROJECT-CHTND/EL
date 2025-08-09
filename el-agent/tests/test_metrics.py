import pytest
from httpx import AsyncClient, ASGITransport

from el_agent.app import app


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_samples():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Touch health to update a gauge
        await ac.get("/health")
        res = await ac.get("/metrics")
        assert res.status_code == 200
        text = res.text
        # Check a couple of metric names exist
        assert "request_latency_seconds_count" in text
        assert "llm_calls_total" in text
        assert "retrieval_calls_total" in text
        assert "hypotheses_open" in text


