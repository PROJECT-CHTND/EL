from __future__ import annotations

import json
import os
from typing import Dict, List

from agent.llm.openai_client import OpenAIClient
from agent.models.question import Question
from agent.prompts.qcheck import SYSTEM_PROMPT, ACCEPT_THRESHOLD
from agent.utils.json_utils import parse_json_strict
from agent.monitoring.trace import trace_event

openai_client = OpenAIClient()


async def return_validated_questions(questions: List[Question]) -> List[Question]:
    """Evaluate questions and return only those meeting threshold criteria."""

    # Serialize input questions for LLM
    q_in = [q.model_dump(exclude_none=True) for q in questions]
    user_content = json.dumps(q_in, ensure_ascii=False)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = await openai_client.call(messages=messages, temperature=0.0, logprobs=False)
    trace_event("stage07_qcheck", "llm_request", {"input_questions": q_in})

    raw_content = response.content or ""
    try:
        arr = parse_json_strict(raw_content)
        if not isinstance(arr, list):
            arr = []
    except Exception:
        arr = []

    # thresholds (configurable)
    spec_th = float(os.getenv("QCHECK_SPEC_THRESHOLD", str(ACCEPT_THRESHOLD)))
    tacit_th = float(os.getenv("QCHECK_TACIT_THRESHOLD", "0.5"))
    max_len = int(os.getenv("QCHECK_MAX_LEN", "240"))

    accepted: List[Question] = []
    for item in arr:
        try:
            q = Question.model_validate(item)
            if (q.specificity or 0.0) >= spec_th and (q.tacit_power or 0.0) >= tacit_th:
                # UX: too long questions are discarded
                if isinstance(q.text, str) and len(q.text) <= max_len:
                    accepted.append(q)
        except Exception:  # noqa: BLE001
            continue

    # Deduplicate by normalized text
    seen = set()
    deduped: List[Question] = []
    for q in accepted:
        key = (q.text or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(q)

    trace_event(
        "stage07_qcheck",
        "validated_questions",
        [q.model_dump(exclude_none=True) for q in deduped],
        meta={"spec_th": spec_th, "tacit_th": tacit_th, "max_len": max_len},
    )
    return deduped