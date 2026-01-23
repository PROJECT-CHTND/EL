"""Evaluation metrics for EL Agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from el_core.llm.client import LLMClient
from el_core.schemas import Insight

from el_eval.schemas import (
    QuestionQuality,
    QuestionQualitySummary,
    SlotMatch,
)

logger = logging.getLogger(__name__)

# Prompt for semantic similarity judgment
SIMILARITY_PROMPT = """以下の2つのテキストが同じ情報を表しているかを判定してください。

## 正解情報（gold）
キー: {slot_key}
値: {slot_value}

## 抽出された情報（extracted）
{insight_content}

## 判定
これらは同じ情報を表していますか？
- 完全に一致または意味的に同等: 1.0
- 部分的に一致（主要な情報は含まれている）: 0.7
- わずかに関連している: 0.3
- 全く関連していない: 0.0

スコアのみを数字で回答してください（例: 0.7）"""


class EvalMetrics:
    """Calculate evaluation metrics for EL Agent performance."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize metrics calculator.
        
        Args:
            llm_client: LLM client for semantic similarity. Creates default if None.
            similarity_threshold: Threshold for considering a slot "matched".
        """
        self._llm = llm_client
        self.similarity_threshold = similarity_threshold

    async def calculate_slot_coverage(
        self,
        gold_slots: dict[str, Any],
        insights: list[Insight] | list[dict[str, Any]],
    ) -> tuple[float, list[SlotMatch], list[str]]:
        """Calculate slot coverage score.
        
        Args:
            gold_slots: Expected slot values from gold_slots.json.
            insights: Insights saved by the agent during conversation.
            
        Returns:
            Tuple of (coverage_score, matched_slots, unmatched_slot_keys).
        """
        # Flatten gold_slots for comparison
        flat_slots = self._flatten_slots(gold_slots)
        
        # Convert insights to comparable format
        insight_contents = self._extract_insight_contents(insights)
        
        matched_slots: list[SlotMatch] = []
        unmatched_slots: list[str] = []
        
        for slot_key, slot_value in flat_slots.items():
            # Find best matching insight
            best_match = await self._find_best_match(
                slot_key, str(slot_value), insight_contents
            )
            
            if best_match:
                matched_slots.append(best_match)
            else:
                unmatched_slots.append(slot_key)
        
        # Calculate coverage
        total_slots = len(flat_slots)
        if total_slots == 0:
            coverage = 1.0
        else:
            coverage = len(matched_slots) / total_slots
        
        logger.info(
            f"Slot coverage: {coverage:.2%} "
            f"({len(matched_slots)}/{total_slots} slots matched)"
        )
        
        return coverage, matched_slots, unmatched_slots

    def _flatten_slots(self, obj: Any, prefix: str = "") -> dict[str, str]:
        """Flatten nested dict into dot-notation keys."""
        result: dict[str, str] = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                result.update(self._flatten_slots(value, new_key))
        elif isinstance(obj, list):
            # For lists, we just stringify the whole thing
            # Individual items are often complex and hard to match individually
            if prefix:
                result[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            if prefix:
                result[prefix] = str(obj)
        
        return result

    def _extract_insight_contents(
        self,
        insights: list[Insight] | list[dict[str, Any]],
    ) -> list[str]:
        """Extract comparable content from insights."""
        contents: list[str] = []
        
        for insight in insights:
            if isinstance(insight, Insight):
                # Format: "subject predicate object"
                content = f"{insight.subject} {insight.predicate} {insight.object}"
            elif isinstance(insight, dict):
                subject = insight.get("subject", "")
                predicate = insight.get("predicate", "")
                obj = insight.get("object", "")
                content = f"{subject} {predicate} {obj}"
            else:
                content = str(insight)
            
            contents.append(content)
        
        return contents

    async def _find_best_match(
        self,
        slot_key: str,
        slot_value: str,
        insight_contents: list[str],
    ) -> SlotMatch | None:
        """Find the best matching insight for a slot.
        
        Uses LLM-based semantic similarity if available,
        otherwise falls back to simple string matching.
        """
        if not insight_contents:
            return None
        
        best_score = 0.0
        best_content = ""
        
        for content in insight_contents:
            if self._llm:
                score = await self._llm_similarity(slot_key, slot_value, content)
            else:
                score = self._simple_similarity(slot_value, content)
            
            if score > best_score:
                best_score = score
                best_content = content
        
        if best_score >= self.similarity_threshold:
            return SlotMatch(
                slot_key=slot_key,
                slot_value=slot_value,
                insight_content=best_content,
                similarity_score=best_score,
                is_match=True,
            )
        
        return None

    async def _llm_similarity(
        self,
        slot_key: str,
        slot_value: str,
        insight_content: str,
    ) -> float:
        """Calculate semantic similarity using LLM."""
        prompt = SIMILARITY_PROMPT.format(
            slot_key=slot_key,
            slot_value=slot_value,
            insight_content=insight_content,
        )
        
        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            
            # Extract content from ChatCompletionMessage and parse score
            content = response.content or ""
            score_str = content.strip()
            return float(score_str)
        
        except Exception as e:
            logger.warning(f"LLM similarity failed: {e}, falling back to simple")
            return self._simple_similarity(slot_value, insight_content)

    def _simple_similarity(self, slot_value: str, insight_content: str) -> float:
        """Simple string-based similarity (fallback).
        
        Checks if key terms from slot_value appear in insight_content.
        """
        slot_lower = slot_value.lower()
        content_lower = insight_content.lower()
        
        # Check for exact substring match
        if slot_lower in content_lower or content_lower in slot_lower:
            return 0.8
        
        # Check for word overlap
        slot_words = set(slot_lower.split())
        content_words = set(content_lower.split())
        
        if not slot_words:
            return 0.0
        
        overlap = len(slot_words & content_words)
        overlap_ratio = overlap / len(slot_words)
        
        return min(overlap_ratio, 0.7)  # Cap at 0.7 for word overlap

    def calculate_turn_efficiency(
        self,
        turn_count: int,
        slot_coverage: float,
        target_coverage: float = 0.8,
    ) -> float:
        """Calculate turn efficiency score.
        
        Rewards achieving high coverage in fewer turns.
        
        Args:
            turn_count: Number of conversation turns.
            slot_coverage: Achieved slot coverage (0-1).
            target_coverage: Target coverage to consider "successful".
            
        Returns:
            Efficiency score (0-1).
        """
        if slot_coverage < target_coverage:
            # Didn't reach target - low efficiency
            return 0.2 * (slot_coverage / target_coverage)
        
        # Reached target - score based on turns taken
        if turn_count <= 3:
            return 1.0
        elif turn_count <= 5:
            return 0.9
        elif turn_count <= 7:
            return 0.7
        elif turn_count <= 10:
            return 0.5
        else:
            return 0.3


class SimpleMetrics:
    """Simplified metrics without LLM - for fast testing."""

    def __init__(self, similarity_threshold: float = 0.3) -> None:
        self.similarity_threshold = similarity_threshold

    async def calculate_slot_coverage(
        self,
        gold_slots: dict[str, Any],
        insights: list[Insight] | list[dict[str, Any]],
    ) -> tuple[float, list[SlotMatch], list[str]]:
        """Calculate coverage using simple string matching."""
        flat_slots = self._flatten_slots(gold_slots)
        insight_contents = [
            f"{i.get('subject', '')} {i.get('predicate', '')} {i.get('object', '')}"
            if isinstance(i, dict)
            else f"{i.subject} {i.predicate} {i.object}"
            for i in insights
        ]
        
        matched_slots: list[SlotMatch] = []
        unmatched_slots: list[str] = []
        
        all_content = " ".join(insight_contents).lower()
        
        for slot_key, slot_value in flat_slots.items():
            # Simple check: any key terms present?
            slot_lower = str(slot_value).lower()
            
            # Extract key terms (words longer than 2 chars)
            key_terms = [w for w in slot_lower.split() if len(w) > 2]
            
            if key_terms:
                matches = sum(1 for t in key_terms if t in all_content)
                score = matches / len(key_terms)
            else:
                score = 0.0
            
            if score >= self.similarity_threshold:
                matched_slots.append(SlotMatch(
                    slot_key=slot_key,
                    slot_value=slot_value,
                    insight_content="[aggregated]",
                    similarity_score=score,
                    is_match=True,
                ))
            else:
                unmatched_slots.append(slot_key)
        
        total = len(flat_slots)
        coverage = len(matched_slots) / total if total > 0 else 1.0
        
        return coverage, matched_slots, unmatched_slots

    def _flatten_slots(self, obj: Any, prefix: str = "") -> dict[str, str]:
        """Flatten nested dict."""
        result: dict[str, str] = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                result.update(self._flatten_slots(value, new_key))
        elif isinstance(obj, list):
            if prefix:
                result[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            if prefix:
                result[prefix] = str(obj)
        
        return result

    def calculate_turn_efficiency(
        self,
        turn_count: int,
        slot_coverage: float,
        target_coverage: float = 0.8,
    ) -> float:
        """Same as EvalMetrics."""
        if slot_coverage < target_coverage:
            return 0.2 * (slot_coverage / target_coverage)
        
        if turn_count <= 3:
            return 1.0
        elif turn_count <= 5:
            return 0.9
        elif turn_count <= 7:
            return 0.7
        elif turn_count <= 10:
            return 0.5
        else:
            return 0.3


# Prompt for LLM-as-Judge question quality evaluation
QUESTION_QUALITY_PROMPT = """あなたはインタビュー品質を評価する審査員です。
以下のインタビュアーの質問を、4つの観点から0.0〜1.0のスコアで評価してください。

## 評価対象の質問
{question}

## 会話の文脈
{context}

## 評価観点

### 1. 共感度 (empathy_score)
- 相手の気持ちや状況を理解しようとしているか
- 寄り添う姿勢が感じられるか
- 0.0: 機械的・冷たい、1.0: 非常に温かく共感的

### 2. 洞察度 (insight_score)
- 単なる情報収集ではなく、深い理解につながる質問か
- 相手が考えを整理したり、新しい気づきを得られる質問か
- 0.0: 表面的・テンプレート的、1.0: 非常に洞察に富む

### 3. 具体性 (specificity_score)
- 曖昧ではなく、具体的な回答を引き出せる質問か
- 適切に絞り込まれているか
- 0.0: 非常に曖昧、1.0: 非常に具体的

### 4. 流れ (flow_score)
- 前の文脈から自然につながっているか
- 会話の流れを壊していないか
- 0.0: 唐突・不自然、1.0: 非常に自然

## 出力形式（JSON）
必ず以下のJSON形式で回答してください：
```json
{{
  "empathy_score": 0.0-1.0,
  "insight_score": 0.0-1.0,
  "specificity_score": 0.0-1.0,
  "flow_score": 0.0-1.0,
  "reasoning": "評価理由を1-2文で"
}}
```"""


class QuestionQualityJudge:
    """LLM-as-Judge for evaluating question quality."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialize the judge.
        
        Args:
            llm_client: LLM client for evaluation. Creates default if None.
        """
        self._llm = llm_client or LLMClient()

    async def evaluate_question(
        self,
        question: str,
        context: str,
        turn_number: int,
    ) -> QuestionQuality:
        """Evaluate a single question's quality.
        
        Args:
            question: The question to evaluate.
            context: Previous conversation context.
            turn_number: Which turn this question appeared in.
            
        Returns:
            QuestionQuality with scores and reasoning.
        """
        prompt = QUESTION_QUALITY_PROMPT.format(
            question=question,
            context=context if context else "(会話の開始)",
        )
        
        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )
            
            # Extract content from ChatCompletionMessage and parse scores
            content = response.content or ""
            scores = self._parse_scores(content)
            
            # Calculate overall score (weighted average)
            overall = (
                scores["empathy_score"] * 0.3 +
                scores["insight_score"] * 0.3 +
                scores["specificity_score"] * 0.2 +
                scores["flow_score"] * 0.2
            )
            
            return QuestionQuality(
                question=question,
                turn_number=turn_number,
                empathy_score=scores["empathy_score"],
                insight_score=scores["insight_score"],
                specificity_score=scores["specificity_score"],
                flow_score=scores["flow_score"],
                overall_score=overall,
                reasoning=scores.get("reasoning", ""),
            )
        
        except Exception as e:
            logger.warning(f"Question quality evaluation failed: {e}")
            # Return neutral scores on failure
            return QuestionQuality(
                question=question,
                turn_number=turn_number,
                empathy_score=0.5,
                insight_score=0.5,
                specificity_score=0.5,
                flow_score=0.5,
                overall_score=0.5,
                reasoning=f"Evaluation failed: {e}",
            )

    def _parse_scores(self, response: str) -> dict[str, Any]:
        """Parse scores from LLM response."""
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return {
                    "empathy_score": float(data.get("empathy_score", 0.5)),
                    "insight_score": float(data.get("insight_score", 0.5)),
                    "specificity_score": float(data.get("specificity_score", 0.5)),
                    "flow_score": float(data.get("flow_score", 0.5)),
                    "reasoning": str(data.get("reasoning", "")),
                }
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Fallback: try to find individual scores
        scores: dict[str, Any] = {"reasoning": ""}
        for key in ["empathy_score", "insight_score", "specificity_score", "flow_score"]:
            match = re.search(rf'{key}["\s:]+([0-9.]+)', response)
            if match:
                scores[key] = min(1.0, max(0.0, float(match.group(1))))
            else:
                scores[key] = 0.5
        
        return scores

    async def evaluate_conversation(
        self,
        conversation_log: list[dict[str, Any]],
    ) -> QuestionQualitySummary:
        """Evaluate all questions in a conversation.
        
        Args:
            conversation_log: List of conversation turns with user_message and assistant_response.
            
        Returns:
            QuestionQualitySummary with all question evaluations.
        """
        questions: list[QuestionQuality] = []
        context = ""
        
        for turn in conversation_log:
            turn_number = turn.get("turn_number", 0)
            assistant_response = turn.get("assistant_response", "")
            user_message = turn.get("user_message", "")
            
            # Skip if no response
            if not assistant_response:
                continue
            
            # Evaluate the assistant's question
            quality = await self.evaluate_question(
                question=assistant_response,
                context=context,
                turn_number=turn_number,
            )
            questions.append(quality)
            
            # Build context for next turn
            context += f"\nUser: {user_message}\nAgent: {assistant_response}"
            # Limit context length
            if len(context) > 2000:
                context = context[-2000:]
        
        # Calculate averages
        if questions:
            avg_empathy = sum(q.empathy_score for q in questions) / len(questions)
            avg_insight = sum(q.insight_score for q in questions) / len(questions)
            avg_specificity = sum(q.specificity_score for q in questions) / len(questions)
            avg_flow = sum(q.flow_score for q in questions) / len(questions)
            avg_overall = sum(q.overall_score for q in questions) / len(questions)
        else:
            avg_empathy = avg_insight = avg_specificity = avg_flow = avg_overall = 0.0
        
        return QuestionQualitySummary(
            avg_empathy=avg_empathy,
            avg_insight=avg_insight,
            avg_specificity=avg_specificity,
            avg_flow=avg_flow,
            avg_overall=avg_overall,
            question_count=len(questions),
            questions=questions,
        )
