from __future__ import annotations

import math
from typing import List

from ..schemas import Hypothesis, StrategistAction


def entropy(p: float) -> float:
    # Binary entropy (bits). Clamp for numerical stability.
    eps = 1e-9
    p = max(eps, min(1 - eps, float(p)))
    return float(-p * math.log2(p) - (1 - p) * math.log2(1 - p))


def p_user_can_answer(h: Hypothesis) -> float:
    # Stub: constant baseline
    return 0.6


def retrievability(h: Hypothesis) -> float:
    # Stub: constant baseline
    return 0.7


def slot_coverage_gain(h: Hypothesis) -> float:
    # Normalize number of slots into [0,1] with soft cap at 5 slots
    return min(1.0, len(h.slots) / 5.0)


class Strategist:
    def __init__(self, tau_stop: float = 0.08) -> None:
        self.tau_stop = float(tau_stop)

    def pick_action(self, h: Hypothesis) -> StrategistAction:
        u = entropy(h.belief)
        ans = p_user_can_answer(h)
        cov = slot_coverage_gain(h)
        voi_ask = (u * ans * cov) / max(h.action_cost.get("ask", 1.0), 1e-6)
        voi_search = (u * retrievability(h) * cov) / max(h.action_cost.get("search", 0.5), 1e-6)

        if max(voi_ask, voi_search) < self.tau_stop:
            return StrategistAction(
                target_hypothesis=h.id,
                action="none",
                question=None,
                expected_gain=0.0,
                estimated_cost=0.0,
                stop_rule_hit=True,
            )

        if voi_ask >= voi_search:
            q = f"{h.text} を確認するための最小質問を1文で。"
            return StrategistAction(
                target_hypothesis=h.id,
                action="ask",
                question=q,
                expected_gain=float(voi_ask),
                estimated_cost=float(h.action_cost.get("ask", 1.0)),
                stop_rule_hit=False,
            )

        return StrategistAction(
            target_hypothesis=h.id,
            action="search",
            question=None,
            expected_gain=float(voi_search),
            estimated_cost=float(h.action_cost.get("search", 0.5)),
            stop_rule_hit=False,
        )

    # Backward-compat plan method (not used in tests)
    def plan(self, goal: str) -> List[StrategistAction]:
        del goal
        # Return a no-op action to keep callers tolerant
        return []


