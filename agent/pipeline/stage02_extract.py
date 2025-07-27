from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from agent.llm.openai_client import OpenAIClient
from agent.models.kg import KGPayload
from agent.kg.client import Neo4jClient
from agent.prompts.extract import SYSTEM_PROMPT, SCHEMA_SNIPPET  # noqa: F401
from agent.llm.schemas import SAVE_KV_FUNCTION


openai_client = OpenAIClient()


async def extract_knowledge(
    text: str,
    *,
    focus: Optional[str] = None,
    temperature: float = 0.0,
) -> KGPayload:
    """Call LLM to extract knowledge graph fragment from the input text.

    Currently this implementation relies on plain JSON-mode extraction. In the
    future, it will switch to function-calling with strict schema validation.
    """

    user_prompt = f"## Input\n{text}\n## Required schema\n{SCHEMA_SNIPPET}"
    if focus:
        user_prompt += f"\n## Focus\n{focus}"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = await openai_client.call(
        messages=messages,
        temperature=temperature,
        functions=[SAVE_KV_FUNCTION],
        function_call={"name": "save_kv"},
        logprobs=False,
    )

    # Parse response depending on whether function_call was used
    if getattr(response, "function_call", None):
        # Function calling path
        try:
            data: Dict[str, Any] = json.loads(response.function_call.arguments)
        except (json.JSONDecodeError, AttributeError):
            data = {"entities": [], "relations": []}
    else:
        raw_content = response.content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.lstrip("`json\n").rstrip("`")
        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            data = {"entities": [], "relations": []}

    payload = KGPayload.model_validate(data)

    if os.getenv("AUTO_INGEST_NEO4J") == "1":
        try:
            # Consider reusing a client instance or making this async
            neo4j_client = Neo4jClient()
            neo4j_client.ingest(payload)
        except Exception as e:
            # Log the error but don't fail the extraction
            # You may want to use your logging framework here
            print(f"Warning: Neo4j ingestion failed: {e}")
    return payload 