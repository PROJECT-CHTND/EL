"""User simulator for evaluation - generates responses based on gold_slots."""

from __future__ import annotations

import json
import logging
from typing import Any

from el_core.llm.client import LLMClient

logger = logging.getLogger(__name__)

# System prompt for the user simulator
SIMULATOR_PROMPT = """あなたは評価用のユーザーシミュレーターです。
以下の「正解情報」を持っている人間として、インタビュアーの質問に自然に回答してください。

## ルール

1. 質問された内容に関連する情報のみを回答してください
2. 一度に全ての情報を出さず、聞かれたことだけに答えてください
3. 自然な日本語で、人間らしく回答してください
4. 正解情報にない内容は「わかりません」「覚えていません」と答えてください
5. 感情的なニュアンスを含めても構いません（例：「大変でした」「焦りました」）

## 正解情報

```json
{gold_slots}
```

## インタビュアーからの質問

{question}

## 回答（自然な日本語で）"""


class UserSimulator:
    """Simulate user responses based on gold_slots data."""

    def __init__(
        self,
        gold_slots: dict[str, Any],
        llm_client: LLMClient | None = None,
        max_info_per_turn: int = 2,
    ) -> None:
        """Initialize the user simulator.
        
        Args:
            gold_slots: The gold standard data to use for responses.
            llm_client: LLM client for generating responses. Creates default if None.
            max_info_per_turn: Maximum pieces of information to reveal per turn.
        """
        self.gold_slots = gold_slots
        self._llm = llm_client or LLMClient()
        self.max_info_per_turn = max_info_per_turn
        
        # Track conversation state
        self._revealed_info: set[str] = set()
        self._turn_count = 0
        self._last_question: str = ""
    
    @property
    def last_question(self) -> str:
        """Get the last question asked by the interviewer."""
        return self._last_question
    
    @property
    def is_complete(self) -> bool:
        """Check if all information has been revealed.
        
        Note: This is a heuristic - we consider complete if we've had
        enough turns to potentially cover all major slots.
        """
        # Estimate based on gold_slots structure
        estimated_slots = self._count_slots(self.gold_slots)
        max_turns_needed = max(estimated_slots // self.max_info_per_turn, 3)
        return self._turn_count >= max_turns_needed

    def _count_slots(self, obj: Any, count: int = 0) -> int:
        """Count the number of leaf values in gold_slots."""
        if isinstance(obj, dict):
            for value in obj.values():
                count = self._count_slots(value, count)
        elif isinstance(obj, list):
            for item in obj:
                count = self._count_slots(item, count)
        else:
            count += 1
        return count

    async def generate_reply(self, question: str) -> str:
        """Generate a user reply to the interviewer's question.
        
        Args:
            question: The interviewer's question.
            
        Returns:
            A natural language response based on gold_slots.
        """
        self._last_question = question
        self._turn_count += 1
        
        # Build the prompt
        prompt = SIMULATOR_PROMPT.format(
            gold_slots=json.dumps(self.gold_slots, ensure_ascii=False, indent=2),
            question=question,
        )
        
        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Some variability in responses
                max_tokens=500,
            )
            
            # Extract content from ChatCompletionMessage
            content = response.content or ""
            
            logger.debug(f"Simulator turn {self._turn_count}: Q='{question[:50]}...' A='{content[:50]}...'")
            return content
        
        except Exception as e:
            logger.error(f"Failed to generate simulator response: {e}")
            # Fallback to a simple response
            return self._fallback_response(question)

    def _fallback_response(self, question: str) -> str:
        """Generate a simple fallback response without LLM.
        
        Used when LLM call fails.
        """
        # Extract a relevant piece of information from gold_slots
        summary = self.gold_slots.get("summary", "")
        if summary:
            return summary[:200] + "..."
        
        # Just acknowledge the question
        return "はい、その件についてお話しします。"

    def reset(self) -> None:
        """Reset the simulator state for a new conversation."""
        self._revealed_info.clear()
        self._turn_count = 0
        self._last_question = ""


class DeterministicSimulator:
    """A simpler simulator that doesn't use LLM - for faster testing.
    
    Returns information from gold_slots in a structured way without
    natural language generation.
    """

    def __init__(self, gold_slots: dict[str, Any]) -> None:
        """Initialize with gold_slots data."""
        self.gold_slots = gold_slots
        self._turn_count = 0
        self._last_question = ""
        self._revealed_keys: set[str] = set()

    @property
    def last_question(self) -> str:
        return self._last_question

    @property
    def is_complete(self) -> bool:
        return self._turn_count >= 10  # Fixed max turns

    def _get_next_unrevealed_info(self) -> tuple[str, Any] | None:
        """Get the next piece of unrevealed information."""
        for key, value in self.gold_slots.items():
            if key not in self._revealed_keys:
                self._revealed_keys.add(key)
                return key, value
        return None

    async def generate_reply(self, question: str) -> str:
        """Generate a structured response.
        
        Returns gold_slots information as JSON-like text.
        """
        self._last_question = question
        self._turn_count += 1
        
        # Get next piece of information
        info = self._get_next_unrevealed_info()
        if info is None:
            return "すべての情報をお伝えしました。"
        
        key, value = info
        if isinstance(value, dict):
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
            return f"{key}について: {value_str}"
        elif isinstance(value, list):
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
            return f"{key}について: {value_str}"
        else:
            return f"{key}は「{value}」です。"

    def reset(self) -> None:
        """Reset state."""
        self._turn_count = 0
        self._last_question = ""
        self._revealed_keys.clear()
