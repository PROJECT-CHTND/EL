from __future__ import annotations

import os
from typing import Any, List, Dict, Optional

import openai


class OpenAIClient:
    """Lightweight wrapper around OpenAI ChatCompletion API.

    This class provides an async `call` interface tailored for function-calling
    and JSON-mode outputs. Retries, logging, and advanced options will be added
    later as needed.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def call(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
    ) -> Any:
        """Invoke the ChatCompletion endpoint and return the first choice."""

        try:
            response = await openai.ChatCompletion.acreate(  # type: ignore[attr-defined]
                model=self.model,
                messages=messages,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                logprobs=logprobs,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

        if not response.choices:
            raise RuntimeError("OpenAI API returned no choices")

        choice = response.choices[0]

        # Audit log (best-effort)
        try:
            import json, datetime, pathlib  # noqa: WPS433, E401

            log_dir = pathlib.Path("logs")
            log_dir.mkdir(exist_ok=True)
            record = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "model": self.model,
                "messages": messages[-2:],  # last user/system pair
                "response": getattr(choice, "content", "function_call"),
            }
            with (log_dir / "llm_calls.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass

        return choice