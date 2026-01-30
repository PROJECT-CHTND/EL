"""Test runner for Eager Learner evaluation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from el_core.agent import ELAgent
from el_core.llm.client import LLMClient
from el_core.schemas import Domain
from el_core.stores.kg_store import KnowledgeGraphStore

from el_eval.loader import DataLoader
from el_eval.metrics import EvalMetrics, QuestionQualityJudge, SimpleMetrics
from el_eval.schemas import (
    ConversationLog,
    EvalDomain,
    EvalResult,
    EvalSummary,
    TestCase,
)
from el_eval.simulator import DeterministicSimulator, UserSimulator

logger = logging.getLogger(__name__)

# Maximum turns per evaluation
MAX_TURNS = 10

# Map from EvalDomain to el_core Domain
DOMAIN_MAP = {
    EvalDomain.POSTMORTEM: Domain.POSTMORTEM,
    EvalDomain.RECIPE: Domain.RECIPE,
    EvalDomain.MANUAL: Domain.DAILY_WORK,  # Map manual/SOP to daily_work
    EvalDomain.DAILY_WORK: Domain.DAILY_WORK,
}


class TestRunner:
    """Run evaluation tests against the EL Agent."""

    def __init__(
        self,
        agent: ELAgent | None = None,
        llm_client: LLMClient | None = None,
        kg_store: KnowledgeGraphStore | None = None,
        use_llm_simulator: bool = True,
        use_llm_metrics: bool = True,
        evaluate_question_quality: bool = False,
        max_turns: int = MAX_TURNS,
    ) -> None:
        """Initialize the test runner.
        
        Args:
            agent: ELAgent instance. Creates default if None.
            llm_client: LLM client for simulator and metrics.
            kg_store: Knowledge graph store for agent.
            use_llm_simulator: Use LLM-based user simulator (slower but more natural).
            use_llm_metrics: Use LLM-based semantic similarity for metrics.
            evaluate_question_quality: Use LLM-as-Judge for question quality (Phase 2).
            max_turns: Maximum conversation turns per test case.
        """
        self._llm = llm_client or LLMClient()
        self._kg_store = kg_store
        self._agent = agent or ELAgent(llm_client=self._llm, kg_store=kg_store)
        
        self.use_llm_simulator = use_llm_simulator
        self.use_llm_metrics = use_llm_metrics
        self.evaluate_question_quality = evaluate_question_quality
        self.max_turns = max_turns
        
        # Initialize metrics calculator
        if use_llm_metrics:
            self._metrics = EvalMetrics(llm_client=self._llm)
        else:
            self._metrics = SimpleMetrics()
        
        # Initialize question quality judge (Phase 2)
        self._quality_judge = QuestionQualityJudge(llm_client=self._llm) if evaluate_question_quality else None

    async def run_case(self, case: TestCase) -> EvalResult:
        """Run evaluation on a single test case.
        
        Args:
            case: The test case to evaluate.
            
        Returns:
            EvalResult with metrics and logs.
        """
        started_at = datetime.now()
        logger.info(f"Starting evaluation for case: {case.case_id} (domain: {case.domain.value})")
        
        # Initialize simulator
        if self.use_llm_simulator:
            simulator = UserSimulator(case.gold_slots, llm_client=self._llm)
        else:
            simulator = DeterministicSimulator(case.gold_slots)
        
        # Start session with initial note
        session_id, opening_message = await self._agent.start_session(
            user_id="eval",
            topic=case.initial_note,
        )
        
        conversation_log: list[ConversationLog] = []
        all_insights: list[dict[str, Any]] = []
        detected_domain = Domain.GENERAL
        
        # Log opening
        conversation_log.append(ConversationLog(
            turn_number=0,
            user_message=case.initial_note,
            assistant_response=opening_message,
        ))
        
        # Run conversation
        current_message = case.initial_note
        
        for turn in range(1, self.max_turns + 1):
            try:
                # Agent responds
                response = await self._agent.respond(session_id, current_message)
                
                # Track insights
                for insight in response.insights_saved:
                    all_insights.append({
                        "subject": insight.subject,
                        "predicate": insight.predicate,
                        "object": insight.object,
                        "confidence": insight.confidence,
                        "domain": insight.domain.value,
                    })
                
                # Track domain
                detected_domain = response.detected_domain
                
                # Log turn
                conversation_log.append(ConversationLog(
                    turn_number=turn,
                    user_message=current_message,
                    assistant_response=response.message,
                    insights_saved=[i.__dict__ if hasattr(i, '__dict__') else i 
                                   for i in response.insights_saved],
                ))
                
                # Check if simulator is done
                if simulator.is_complete:
                    logger.info(f"Case {case.case_id}: Simulator indicates completion at turn {turn}")
                    break
                
                # Generate next user message
                current_message = await simulator.generate_reply(response.message)
                
            except Exception as e:
                logger.error(f"Error in turn {turn} for case {case.case_id}: {e}")
                break
        
        # End session
        self._agent.end_session(session_id)
        
        # Calculate metrics
        coverage, matched_slots, unmatched_slots = await self._metrics.calculate_slot_coverage(
            case.gold_slots,
            all_insights,
        )
        
        turn_count = len(conversation_log) - 1  # Exclude initial turn
        turn_efficiency = self._metrics.calculate_turn_efficiency(turn_count, coverage)
        
        # Check domain accuracy
        expected_domain = DOMAIN_MAP.get(case.domain, Domain.GENERAL)
        domain_accuracy = detected_domain == expected_domain
        
        # Evaluate question quality (Phase 2)
        question_quality = None
        if self._quality_judge:
            logger.info(f"Evaluating question quality for case {case.case_id}...")
            conversation_dicts = [
                {
                    "turn_number": log.turn_number,
                    "user_message": log.user_message,
                    "assistant_response": log.assistant_response,
                }
                for log in conversation_log
            ]
            question_quality = await self._quality_judge.evaluate_conversation(conversation_dicts)
            logger.info(
                f"Question quality: empathy={question_quality.avg_empathy:.2f}, "
                f"insight={question_quality.avg_insight:.2f}, "
                f"overall={question_quality.avg_overall:.2f}"
            )
        
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()
        
        result = EvalResult(
            case_id=case.case_id,
            domain=case.domain,
            slot_coverage=coverage,
            turn_count=turn_count,
            turn_efficiency=turn_efficiency,
            domain_accuracy=domain_accuracy,
            question_quality=question_quality,
            matched_slots=matched_slots,
            unmatched_slots=unmatched_slots,
            conversation_log=conversation_log,
            insights_saved=all_insights,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
        )
        
        logger.info(
            f"Completed case {case.case_id}: "
            f"coverage={coverage:.2%}, turns={turn_count}, "
            f"efficiency={turn_efficiency:.2f}, domain_ok={domain_accuracy}"
            + (f", quality={question_quality.avg_overall:.2f}" if question_quality else "")
        )
        
        return result

    async def run_all(
        self,
        data_loader: DataLoader,
        domain_filter: EvalDomain | None = None,
        case_filter: str | None = None,
        parallel: bool = False,
    ) -> EvalSummary:
        """Run evaluation on all test cases.
        
        Args:
            data_loader: DataLoader instance.
            domain_filter: Only run cases from this domain.
            case_filter: Only run this specific case.
            parallel: Run cases in parallel (faster but may hit rate limits).
            
        Returns:
            EvalSummary with aggregate results.
        """
        started_at = datetime.now()
        
        # Load cases
        cases = list(data_loader.iter_cases(domain=domain_filter, case_id=case_filter))
        logger.info(f"Running evaluation on {len(cases)} test cases")
        
        results: list[EvalResult] = []
        failed_cases = 0
        
        if parallel:
            # Run in parallel
            tasks = [self.run_case(case) for case in cases]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_results = []
            for r in results:
                if isinstance(r, Exception):
                    logger.error(f"Case failed with exception: {r}")
                    failed_cases += 1
                else:
                    valid_results.append(r)
            results = valid_results
        else:
            # Run sequentially
            for case in cases:
                try:
                    result = await self.run_case(case)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Case {case.case_id} failed: {e}")
                    failed_cases += 1
        
        completed_at = datetime.now()
        
        # Calculate aggregate metrics
        if results:
            avg_coverage = sum(r.slot_coverage for r in results) / len(results)
            avg_turns = sum(r.turn_count for r in results) / len(results)
            avg_efficiency = sum(r.turn_efficiency for r in results) / len(results)
            domain_accuracy_rate = sum(1 for r in results if r.domain_accuracy) / len(results)
        else:
            avg_coverage = avg_turns = avg_efficiency = domain_accuracy_rate = 0.0
        
        # Calculate question quality aggregates (Phase 2)
        avg_question_quality = None
        avg_empathy = None
        avg_insight = None
        avg_specificity = None
        
        results_with_quality = [r for r in results if r.question_quality is not None]
        if results_with_quality:
            avg_question_quality = sum(r.question_quality.avg_overall for r in results_with_quality) / len(results_with_quality)
            avg_empathy = sum(r.question_quality.avg_empathy for r in results_with_quality) / len(results_with_quality)
            avg_insight = sum(r.question_quality.avg_insight for r in results_with_quality) / len(results_with_quality)
            avg_specificity = sum(r.question_quality.avg_specificity for r in results_with_quality) / len(results_with_quality)
        
        # Per-domain breakdown
        by_domain: dict[str, dict[str, float]] = {}
        for domain in EvalDomain:
            domain_results = [r for r in results if r.domain == domain]
            if domain_results:
                domain_stats: dict[str, float] = {
                    "count": len(domain_results),
                    "avg_coverage": sum(r.slot_coverage for r in domain_results) / len(domain_results),
                    "avg_turns": sum(r.turn_count for r in domain_results) / len(domain_results),
                    "avg_efficiency": sum(r.turn_efficiency for r in domain_results) / len(domain_results),
                }
                # Add quality metrics if available
                domain_quality_results = [r for r in domain_results if r.question_quality is not None]
                if domain_quality_results:
                    domain_stats["avg_quality"] = sum(r.question_quality.avg_overall for r in domain_quality_results) / len(domain_quality_results)
                by_domain[domain.value] = domain_stats
        
        return EvalSummary(
            total_cases=len(cases),
            completed_cases=len(results),
            failed_cases=failed_cases,
            avg_slot_coverage=avg_coverage,
            avg_turn_count=avg_turns,
            avg_turn_efficiency=avg_efficiency,
            domain_accuracy_rate=domain_accuracy_rate,
            avg_question_quality=avg_question_quality,
            avg_empathy=avg_empathy,
            avg_insight=avg_insight,
            avg_specificity=avg_specificity,
            by_domain=by_domain,
            results=results,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            started_at=started_at,
            completed_at=completed_at,
        )
