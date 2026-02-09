"""LLM Client for GPT-5.2 with tool calling support and multi-tier model selection."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client with multi-tier model support.

    Three model tiers are available:
    - MODEL_FLAGSHIP (gpt-5.2): Main conversation responses visible to users.
    - MODEL_MID (gpt-5-mini): Structured analysis tasks (consistency checks, extraction).
    - MODEL_FAST (gpt-5-nano): Simple classification / tagging tasks.
    """

    DEFAULT_MODEL = "gpt-5.2"

    # --- Model tier constants (resolved from env at class-load time) ---
    MODEL_FLAGSHIP: str = os.getenv("OPENAI_MODEL", "gpt-5.2")
    MODEL_MID: str = os.getenv("OPENAI_MODEL_MID", "gpt-5-mini")
    MODEL_FAST: str = os.getenv("OPENAI_MODEL_FAST", "gpt-5-nano")

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            model: Model name. Defaults to gpt-5.2.
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> ChatCompletionMessage:
        """Send a chat completion request.

        Args:
            messages: List of messages in OpenAI format.
            model: Model override for this call. Falls back to instance default.
            tools: Optional list of tools available to the model.
            tool_choice: How to select tools ("auto", "none", or specific tool).
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature. None lets the model decide.

        Returns:
            The assistant's response message.
        """
        resolved_model = model or self._model
        # GPT-5 uses max_completion_tokens instead of max_tokens
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        # Some models (like gpt-5.x) may not support temperature parameter
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message
        except Exception as e:
            error_msg = str(e)
            # Retry without temperature if unsupported
            if "temperature" in error_msg.lower() and "unsupported" in error_msg.lower():
                logger.warning("Temperature not supported by model, retrying without it")
                kwargs.pop("temperature", None)
                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message
            raise

    async def chat_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        tool_handlers: dict[str, Any],
        *,
        model: str | None = None,
        max_tool_rounds: int = 5,
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Chat with automatic tool execution.

        Args:
            messages: List of messages.
            tools: List of available tools.
            tool_handlers: Dict mapping tool names to handler functions.
            model: Model override for this call. Falls back to instance default.
            max_tool_rounds: Maximum number of tool calling rounds.
            max_tokens: Maximum tokens per response.

        Returns:
            Tuple of (final_message, list_of_tool_results).
        """
        current_messages = list(messages)
        all_tool_results: list[dict[str, Any]] = []

        for _ in range(max_tool_rounds):
            response = await self.chat(
                current_messages,
                model=model,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
            )

            # If no tool calls, return the content
            if not response.tool_calls:
                return response.content or "", all_tool_results

            # Process tool calls
            current_messages.append(
                {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}  # type: ignore
            )

            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                if tool_name in tool_handlers:
                    try:
                        handler = tool_handlers[tool_name]
                        if callable(handler):
                            # Check if handler is async
                            import asyncio

                            if asyncio.iscoroutinefunction(handler):
                                result = await handler(**tool_args)
                            else:
                                result = handler(**tool_args)
                        else:
                            result = {"error": f"Handler for {tool_name} is not callable"}
                    except Exception as e:
                        logger.error(f"Tool {tool_name} failed: {e}")
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_result = {
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": result,
                }
                all_tool_results.append(tool_result)

                # Add tool result to messages
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }  # type: ignore
                )

        # Max rounds reached, get final response without tools
        final_response = await self.chat(
            current_messages,
            model=model,
            tools=None,
            max_tokens=max_tokens,
        )
        return final_response.content or "", all_tool_results

    @staticmethod
    def create_tool_definition(
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> ChatCompletionToolParam:
        """Create a tool definition in OpenAI format.

        Args:
            name: Tool name.
            description: Tool description.
            parameters: JSON Schema for parameters.

        Returns:
            Tool definition in OpenAI format.
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
