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


# --- High-level one-turn orchestrator (sequential) ---
from typing import Optional  # noqa: E402
from agent.pipeline.stage02_extract import extract_knowledge  # noqa: E402
from agent.pipeline.stage03_merge import merge_and_persist  # noqa: E402
from agent.pipeline.stage04_slots import propose_slots  # noqa: E402
from agent.pipeline.stage06_qgen import generate_questions  # noqa: E402
from agent.pipeline.stage07_qcheck import return_validated_questions  # noqa: E402
from agent.models.question import Question  # noqa: E402


async def run_turn(
    answer_text: str,
    *,
    topic_meta: Optional[str] = None,
    max_slots: int = 3,
    max_questions: int = 10,
) -> list[Question]:
    """Execute Stage02→03→04→06→07 sequentially and return validated questions.

    Non-fatal behavior:
      - Stage03(merge) failures do not abort the turn.
    """

    # Stage02 – extract KG fragment
    kg = await extract_knowledge(answer_text, focus=topic_meta)

    # Stage03 – merge/persist KG (non-blocking)
    try:
        merge_and_persist(kg)
    except Exception as _:
        # swallow merge errors; proceed with the best-effort KG
        pass

    # Stage04 – propose slots
    slots = await propose_slots(kg, topic_meta=topic_meta, max_slots=max_slots)
    if not slots:
        return []

    # Stage06 – generate questions
    questions = await generate_questions(slots, max_questions=max_questions)
    if not questions:
        return []

    # Stage07 – validate questions
    validated = await return_validated_questions(questions)
    return validated