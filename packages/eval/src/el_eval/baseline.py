"""Baseline agents for comparison with Eager Learner."""

from __future__ import annotations

import logging
from typing import Any

from el_core.llm.client import LLMClient

logger = logging.getLogger(__name__)


# Simple template-based prompts for baseline comparison
SIMPLE_INTERVIEWER_PROMPT = """あなたはインタビュアーです。
ユーザーから「{topic}」について話を聞いています。

必要な情報を収集するために質問してください。
一度に1つの質問だけをしてください。"""


FORM_FILLER_PROMPT = """あなたは情報収集アシスタントです。
ユーザーから「{topic}」について情報を収集しています。

以下の項目を埋めるために、順番に質問してください：
- 概要（summary）
- 影響範囲（impact）
- タイムライン（timeline）
- 根本原因（root_cause）
- 対策（remediation）

まだ聞いていない項目について質問してください。"""


class BaselineAgent:
    """Simple baseline agent for comparison.
    
    Uses a straightforward prompt without the sophisticated
    empathy and insight features of ELAgent.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        prompt_type: str = "simple",
    ) -> None:
        """Initialize baseline agent.
        
        Args:
            llm_client: LLM client. Creates default if None.
            prompt_type: Type of prompt to use ("simple" or "form").
        """
        self._llm = llm_client or LLMClient()
        self.prompt_type = prompt_type
        self._sessions: dict[str, dict[str, Any]] = {}
    
    def _get_system_prompt(self, topic: str) -> str:
        """Get the system prompt based on type."""
        if self.prompt_type == "form":
            return FORM_FILLER_PROMPT.format(topic=topic)
        return SIMPLE_INTERVIEWER_PROMPT.format(topic=topic)

    async def start_session(
        self,
        user_id: str,
        topic: str,
    ) -> tuple[str, str]:
        """Start a session.
        
        Args:
            user_id: User ID.
            topic: Conversation topic.
            
        Returns:
            Tuple of (session_id, opening_message).
        """
        import uuid
        session_id = str(uuid.uuid4())
        
        self._sessions[session_id] = {
            "topic": topic,
            "messages": [],
            "system_prompt": self._get_system_prompt(topic),
        }
        
        # Generate opening question
        response = await self._llm.chat(
            messages=[
                {"role": "system", "content": self._sessions[session_id]["system_prompt"]},
                {"role": "user", "content": f"トピック: {topic}"},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        
        # Extract content from ChatCompletionMessage
        opening = response.content or ""
        
        self._sessions[session_id]["messages"].append(
            {"role": "assistant", "content": opening}
        )
        
        return session_id, opening

    async def respond(
        self,
        session_id: str,
        user_message: str,
    ) -> dict[str, Any]:
        """Generate response to user message.
        
        Args:
            session_id: Session ID.
            user_message: User's message.
            
        Returns:
            Dict with message and empty tool calls/insights.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user message
        session["messages"].append({"role": "user", "content": user_message})
        
        # Build messages for API
        messages = [
            {"role": "system", "content": session["system_prompt"]},
        ] + session["messages"]
        
        # Generate response
        llm_response = await self._llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )
        
        # Extract content from ChatCompletionMessage
        response_text = llm_response.content or ""
        
        session["messages"].append({"role": "assistant", "content": response_text})
        
        # Return in similar format to ELAgent
        return {
            "message": response_text,
            "insights_saved": [],  # Baseline doesn't save insights
            "knowledge_used": [],
            "detected_domain": "general",
        }

    def end_session(self, session_id: str) -> None:
        """End a session."""
        self._sessions.pop(session_id, None)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session data."""
        return self._sessions.get(session_id)


class ComparisonRunner:
    """Run comparison between EL Agent and baseline."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Initialize comparison runner."""
        self._llm = llm_client or LLMClient()

    async def compare_single_case(
        self,
        topic: str,
        gold_slots: dict[str, Any],
        max_turns: int = 5,
    ) -> dict[str, Any]:
        """Compare EL Agent vs baseline on a single case.
        
        Args:
            topic: The initial topic/note.
            gold_slots: Gold slots for reference.
            max_turns: Maximum turns to run.
            
        Returns:
            Comparison results with quality scores for both.
        """
        from el_core.agent import ELAgent
        from el_eval.metrics import QuestionQualityJudge
        from el_eval.simulator import UserSimulator
        
        # Initialize agents
        el_agent = ELAgent(llm_client=self._llm)
        baseline_simple = BaselineAgent(llm_client=self._llm, prompt_type="simple")
        baseline_form = BaselineAgent(llm_client=self._llm, prompt_type="form")
        
        # Initialize quality judge
        judge = QuestionQualityJudge(llm_client=self._llm)
        
        results = {}
        
        for name, agent in [
            ("el_agent", el_agent),
            ("baseline_simple", baseline_simple),
            ("baseline_form", baseline_form),
        ]:
            logger.info(f"Running comparison for {name}...")
            
            # Create fresh simulator for each agent
            simulator = UserSimulator(gold_slots, llm_client=self._llm)
            
            # Start session
            session_id, opening = await agent.start_session("compare", topic)
            
            conversation = [
                {"turn_number": 0, "user_message": topic, "assistant_response": opening}
            ]
            
            current_message = topic
            
            for turn in range(1, max_turns + 1):
                try:
                    # Get agent response
                    response = await agent.respond(session_id, current_message)
                    
                    # Handle different response formats
                    if isinstance(response, dict):
                        response_text = response.get("message", "")
                    else:
                        response_text = response.message
                    
                    conversation.append({
                        "turn_number": turn,
                        "user_message": current_message,
                        "assistant_response": response_text,
                    })
                    
                    # Generate user reply
                    if simulator.is_complete or turn >= max_turns:
                        break
                    current_message = await simulator.generate_reply(response_text)
                    
                except Exception as e:
                    logger.warning(f"Error in {name} turn {turn}: {e}")
                    break
            
            # Evaluate quality
            quality = await judge.evaluate_conversation(conversation)
            
            results[name] = {
                "conversation_length": len(conversation),
                "avg_empathy": quality.avg_empathy,
                "avg_insight": quality.avg_insight,
                "avg_specificity": quality.avg_specificity,
                "avg_flow": quality.avg_flow,
                "avg_overall": quality.avg_overall,
            }
            
            # Clean up
            agent.end_session(session_id)
        
        return results

    def format_comparison(self, results: dict[str, Any]) -> str:
        """Format comparison results as a readable string."""
        lines = ["=" * 60, "Comparison Results", "=" * 60, ""]
        
        # Header
        lines.append(f"{'Agent':<20} {'Empathy':>8} {'Insight':>8} {'Specific':>8} {'Flow':>8} {'Overall':>8}")
        lines.append("-" * 60)
        
        # Results
        for agent_name, scores in results.items():
            display_name = {
                "el_agent": "EL Agent",
                "baseline_simple": "Simple Baseline",
                "baseline_form": "Form Filler",
            }.get(agent_name, agent_name)
            
            lines.append(
                f"{display_name:<20} "
                f"{scores['avg_empathy']:>8.2f} "
                f"{scores['avg_insight']:>8.2f} "
                f"{scores['avg_specificity']:>8.2f} "
                f"{scores['avg_flow']:>8.2f} "
                f"{scores['avg_overall']:>8.2f}"
            )
        
        lines.append("-" * 60)
        
        # Winner
        el_score = results.get("el_agent", {}).get("avg_overall", 0)
        baseline_scores = [
            results.get(k, {}).get("avg_overall", 0)
            for k in ["baseline_simple", "baseline_form"]
        ]
        best_baseline = max(baseline_scores) if baseline_scores else 0
        
        if el_score > best_baseline:
            diff = el_score - best_baseline
            lines.append(f"\n✓ EL Agent wins by {diff:.2f} points")
        elif el_score < best_baseline:
            diff = best_baseline - el_score
            lines.append(f"\n✗ Baseline wins by {diff:.2f} points")
        else:
            lines.append("\n= Tie")
        
        return "\n".join(lines)
