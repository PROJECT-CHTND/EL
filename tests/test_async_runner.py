import asyncio

import pytest

from agent.pipeline.runner import AsyncPipelineRunner, StageSpec


@pytest.mark.asyncio
async def test_async_runner_retries():
    """Stage should succeed after one retry."""

    attempts = {"count": 0}

    async def flaky_stage():
        if attempts["count"] == 0:
            attempts["count"] += 1
            raise RuntimeError("temporary failure")
        return "ok"

    runner = AsyncPipelineRunner(
        stages=[StageSpec(name="flaky", func=flaky_stage, max_retries=2)]
    )

    results = await runner.run()
    assert results["flaky"] == "ok"
    # ensure exactly one retry happened (two total attempts)
    assert attempts["count"] == 1

# ---------------------------------------------------------------------------
# Exhausted retry scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_runner_exhausts_retries(monkeypatch):
    """Stage should raise after exceeding max retries."""

    # Patch asyncio.sleep to avoid real delays during testing
    async def _instant_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    attempts: dict[str, int] = {"count": 0}

    async def always_fail_stage():
        attempts["count"] += 1
        raise RuntimeError("permanent failure")

    max_retries = 2
    runner = AsyncPipelineRunner(
        stages=[StageSpec(name="fail", func=always_fail_stage, max_retries=max_retries)]
    )

    with pytest.raises(RuntimeError):
        await runner.run()

    # There should be one initial attempt plus `max_retries` retries.
    assert attempts["count"] == max_retries + 1 