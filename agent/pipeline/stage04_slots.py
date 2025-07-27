from __future__ import annotations

import json
from typing import Dict, List, Optional

from agent.llm.openai_client import OpenAIClient
from agent.models.kg import KGPayload
from agent.prompts.slots import SYSTEM_PROMPT
from agent.slots import Slot, SlotRegistry

openai_client = OpenAIClient()


async def propose_slots(
    kg: KGPayload,
    *,
    topic_meta: Optional[str] = None,
    registry: Optional[SlotRegistry] = None,
    max_slots: int = 3,
) -> List[Slot]:
    """Analyze KG and propose up to `max_slots` new slots.

    The LLM returns an array of JSON objects. Each object must map to `Slot` schema.
    If a registry is provided, proposed slots are added to it.
    """

    user_content = (
        f"## Current KG\n{kg.model_dump_json()}\n" + (f"## Topic\n{topic_meta}" if topic_meta else "")
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = await openai_client.call(messages=messages, temperature=0.4, logprobs=False)

    raw_content = response.content.strip()
    if raw_content.startswith("```"):
        raw_content = raw_content.lstrip("`json\n").rstrip("`")

    try:
        suggestions = json.loads(raw_content)
        if not isinstance(suggestions, list):
            suggestions = []
    except json.JSONDecodeError:
        suggestions = []

    slots: List[Slot] = []
    for item in suggestions[:max_slots]:
        try:
            slots.append(Slot.model_validate(item))
        except Exception:  # noqa: BLE001
            continue

    if registry is not None:
        for slot in slots:
            registry.add(slot)

    return slots 