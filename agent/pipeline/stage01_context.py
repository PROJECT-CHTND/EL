from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import redis.asyncio as aioredis  # type: ignore

from agent.llm.openai_client import OpenAIClient
from agent.models.context import ContextPayload
from agent.prompts.context import SYSTEM_PROMPT
from agent.utils.json_utils import parse_json_strict

openai_client = OpenAIClient()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_STREAM = os.getenv("CONTEXT_STREAM", "context_stream")


async def generate_context(text: str) -> ContextPayload:
    """Generate 3-level hierarchical summaries and optionally publish to Redis Stream."""

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    response = await openai_client.call(messages=messages, temperature=0.3, logprobs=False)

    raw_content = response.content or ""
    try:
        data: Dict[str, Any] = parse_json_strict(raw_content)
    except Exception:
        data = {"rag_keys": "", "mid_summary": "", "global_summary": ""}

    payload = ContextPayload.model_validate(data)

    # Publish to Redis Stream if available
    if os.getenv("PUBLISH_CONTEXT_STREAM") == "1":
        redis = aioredis.from_url(REDIS_URL)
        await redis.xadd(
            REDIS_STREAM,
            {
                "rag_keys": payload.rag_keys,
                "mid_summary": payload.mid_summary,
                "global_summary": payload.global_summary,
            },
        )
        await redis.aclose()

    return payload 