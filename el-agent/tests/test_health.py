import pytest
from httpx import AsyncClient, ASGITransport

from el_agent.app import app


@pytest.mark.asyncio
async def test_health_ok():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        assert res.json() == {"ok": True}


