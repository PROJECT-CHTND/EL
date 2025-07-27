from __future__ import annotations

import json
from typing import Dict, List

from agent.llm.openai_client import OpenAIClient
from agent.models.question import Question
from agent.prompts.qcheck import SYSTEM_PROMPT, ACCEPT_THRESHOLD

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

    raw_content = response.content.strip()
    if raw_content.startswith("```"):
        raw_content = raw_content.lstrip("`json\n").rstrip("`")

    try:
        arr = json.loads(raw_content)
        if not isinstance(arr, list):
            arr = []
    except json.JSONDecodeError:
        arr = []

    accepted: List[Question] = []
    for item in arr:
        try:
            q = Question.model_validate(item)
            if (q.specificity or 0.0) >= ACCEPT_THRESHOLD and (q.tacit_power or 0.0) >= ACCEPT_THRESHOLD:
                accepted.append(q)
        except Exception:  # noqa: BLE001
            continue

    return accepted 