from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml  # type: ignore

from agent.llm.openai_client import OpenAIClient
from agent.prompts.qgen import SYSTEM_PROMPT
from agent.slots import Slot
from agent.models.question import Question
from agent.monitoring.trace import trace_event
from agent.utils.json_utils import parse_json_strict

openai_client = OpenAIClient()

DEFAULT_STRATEGY_PATH = Path(__file__).parent.parent / "prompts" / "qgen_strategy.yaml"


def _load_strategy_map(path: Path) -> Dict[str, Dict[str, str]]:
    return yaml.safe_load(path.read_text())  # type: ignore[arg-type]


def _render_template(template: str, slot: Slot) -> str:
    return template.replace("{{name}}", slot.name).replace("{{description}}", slot.description)


async def generate_questions(
    slots: List[Slot],
    *,
    strategy_path: Path = DEFAULT_STRATEGY_PATH,
    max_questions: int = 10,
) -> List[Question]:
    """Generate questions for the given slots using strategy map."""

    strategy_map = _load_strategy_map(strategy_path)

    # Build pseudo examples for system prompt using strategy templates
    examples: List[Dict[str, str]] = []
    for slot in slots[:max_questions]:
        key: str = slot.type or "default"
        strat = strategy_map.get(key) or strategy_map.get("default")

        if not strat:
            raise ValueError(f"No strategy found for slot type '{key}' and no default strategy")

        template: str
        if isinstance(strat, dict):
            tmpl = strat.get("template")
            if not isinstance(tmpl, str):
                raise ValueError(f"Strategy for slot type '{key}' missing valid 'template' value")
            template = tmpl
        elif isinstance(strat, str):
            template = strat
        else:
            raise ValueError(f"Invalid strategy format for key '{key}': {type(strat)}")

        examples.append(
            {
                "slot_name": slot.name,
                "question": _render_template(template, slot),
            }
        )

    user_content = json.dumps(examples, ensure_ascii=False)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = await openai_client.call(messages=messages, temperature=0.6, logprobs=False)
    trace_event("stage06_qgen", "llm_request", {"examples": examples})

    raw_content = response.content or ""
    try:
        arr = parse_json_strict(raw_content)
        if not isinstance(arr, list):
            arr = []
    except Exception:
        arr = []

    questions: List[Question] = []
    for item in arr:
        try:
            questions.append(Question.model_validate(item))
        except Exception:  # noqa: BLE001
            continue
    trace_event("stage06_qgen", "generated_questions", [q.model_dump(exclude_none=True) for q in questions])
    return questions