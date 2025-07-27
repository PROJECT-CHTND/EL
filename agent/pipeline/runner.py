from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class StageSpec:
    """Specification for a pipeline stage."""

    name: str
    func: Callable[..., Awaitable[Any]]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 2


class AsyncPipelineRunner:
    """Run multiple pipeline stages concurrently with retry support."""

    def __init__(self, stages: List[StageSpec]):
        self.stages = stages

    async def _run_stage(self, spec: StageSpec) -> Any:
        """Run a single stage with retry logic."""
        attempt = 0
        backoff = 0.5  # seconds
        while True:
            try:
                logger.debug("Running stage %s (attempt %d)", spec.name, attempt + 1)
                result = await spec.func(**spec.kwargs)
                logger.info("Stage %s completed", spec.name)
                return result
            except Exception as exc:  # pylint: disable=broad-except
                attempt += 1
                logger.warning(
                    "Stage %s failed on attempt %d/%d: %s",
                    spec.name,
                    attempt,
                    spec.max_retries,
                    exc,
                )
                if attempt > spec.max_retries:
                    logger.error("Stage %s exceeded max retries", spec.name)
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2  # exponential backoff

    async def run(self) -> Dict[str, Any]:
        """Run all stages in parallel and return a mapping of results."""
        tasks = {spec.name: asyncio.create_task(self._run_stage(spec)) for spec in self.stages}
        try:
            # Await all tasks concurrently
            completed_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
            results: Dict[str, Any] = {}

            for (name, task), result in zip(tasks.items(), completed_tasks):
                if isinstance(result, Exception):
                    logger.error("Stage %s failed: %s", name, result)
                    raise result
                results[name] = result
            return results
        except Exception:
            # Cancel any remaining tasks
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            raise