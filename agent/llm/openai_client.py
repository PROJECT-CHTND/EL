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
    ) -> Dict[str, Any]:
        """Invoke the ChatCompletion endpoint and return the first choice."""
        
        response = await self.client.chat.completions.create(
            response = await openai.ChatCompletion.acreate(  # type: ignore[attr-defined]
                model=self.model,
                messages=messages,
                temperature=temperature,
                functions=functions,
                function_call=function_call,
                logprobs=logprobs,
            )

            if not response.choices:
                raise RuntimeError("OpenAI API returned no choices")

            return response.choices[0]
        except openai.error.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e
            raise RuntimeError("OpenAI API returned no choices")
        return response.choices[0]