"""LLM Client for GPT-5.2 with tool calling support and multi-tier model selection."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
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

    async def chat_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens.

        Yields content delta strings as they arrive from the API.
        """
        resolved_model = model or self._model
        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            error_msg = str(e)
            if "temperature" in error_msg.lower() and "unsupported" in error_msg.lower():
                logger.warning("Temperature not supported by model, retrying without it (stream)")
                kwargs.pop("temperature", None)
                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content
            else:
                raise

    async def chat_with_tools_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        tool_handlers: dict[str, Any],
        *,
        model: str | None = None,
        max_tool_rounds: int = 5,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict[str, Any]]:
        """Chat with tools, streaming every response round.

        Each round streams via the OpenAI API. If the model produces content
        tokens they are yielded immediately. If it produces tool calls, those
        are accumulated from the stream, executed, and the loop continues.

        Yields dicts:
            {"type": "tool_call", "name": ..., "arguments": ..., "result": ...}
            {"type": "token", "content": "..."}
            {"type": "tool_results", "results": [...]}   (emitted once at end)
        """
        import asyncio as _asyncio

        current_messages = list(messages)
        all_tool_results: list[dict[str, Any]] = []
        resolved_model = model or self._model

        for _round in range(max_tool_rounds + 1):
            # Determine whether to provide tools
            use_tools = _round < max_tool_rounds
            kwargs: dict[str, Any] = {
                "model": resolved_model,
                "messages": current_messages,
                "max_completion_tokens": max_tokens,
                "stream": True,
            }
            if use_tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            stream = await self.client.chat.completions.create(**kwargs)

            # Accumulate streamed content and tool-call fragments
            content_parts: list[str] = []
            # tool_calls_acc: {index: {"id": ..., "name": ..., "arguments": ...}}
            tool_calls_acc: dict[int, dict[str, str]] = {}

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue
                delta = choice.delta

                # Content tokens — yield immediately
                if delta.content:
                    content_parts.append(delta.content)
                    yield {"type": "token", "content": delta.content}

                # Tool-call deltas — accumulate
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_acc[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

            # If no tool calls were made, this was the final text round
            if not tool_calls_acc:
                yield {"type": "tool_results", "results": all_tool_results}
                return

            # Build assistant message with accumulated tool calls for the conversation
            from openai.types.chat.chat_completion_message_tool_call import (
                ChatCompletionMessageToolCall,
                Function,
            )
            assembled_tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tc["id"],
                    type="function",
                    function=Function(name=tc["name"], arguments=tc["arguments"]),
                )
                for tc in (tool_calls_acc[i] for i in sorted(tool_calls_acc))
            ]
            full_content = "".join(content_parts) or None
            current_messages.append(
                {"role": "assistant", "content": full_content, "tool_calls": assembled_tool_calls}  # type: ignore
            )

            # Execute each tool call
            for tc_data in (tool_calls_acc[i] for i in sorted(tool_calls_acc)):
                tool_name = tc_data["name"]
                try:
                    tool_args = json.loads(tc_data["arguments"])
                except json.JSONDecodeError:
                    tool_args = {}
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                if tool_name in tool_handlers:
                    try:
                        handler = tool_handlers[tool_name]
                        if callable(handler):
                            if _asyncio.iscoroutinefunction(handler):
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
                    "tool_call_id": tc_data["id"],
                    "name": tool_name,
                    "arguments": tool_args,
                    "result": result,
                }
                all_tool_results.append(tool_result)
                yield {"type": "tool_call", "name": tool_name, "arguments": tool_args, "result": result}

                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_data["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }  # type: ignore
                )

        # Fell through all rounds
        yield {"type": "tool_results", "results": all_tool_results}

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
