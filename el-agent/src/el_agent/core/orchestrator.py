from __future__ import annotations

from typing import List

from .evaluator import Evaluator
from .knowledge_integrator import KnowledgeIntegrator
from .strategist import Strategist
from ..schemas import Evidence, Hypothesis


class Orchestrator:
    def __init__(self) -> None:
        self.strategist = Strategist()
        self.integrator = KnowledgeIntegrator()
        self.evaluator = Evaluator()

    def run(self, goal: str) -> float:
        _plan = self.strategist.plan(goal)
        hypothesis = Hypothesis(hypothesis_id="h-1", text=goal)
        evidences: List[Evidence] = []
        hypothesis = self.integrator.integrate(hypothesis, evidences)
        score = self.evaluator.score(hypothesis, evidences)
        return score


