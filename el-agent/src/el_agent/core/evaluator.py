from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..schemas import Evidence, Hypothesis


def sigmoid(x: float) -> float:
    import math

    # Avoid overflow
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    import math

    # Clamp away from 0/1 for numerical stability
    eps = 1e-9
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1.0 - p))


class ConfidenceEvaluator:
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        default = {
            "cosine": 0.6,
            "source_trust": 0.3,
            "recency": 0.2,
            "logic_ok": 0.4,
            "redundancy_penalty": -0.2,
        }
        self.weights = {**default, **(weights or {})}
        # Scale factor from signed evidence score in [-1, 1] to logit delta
        self.scale = 2.0

    def score(self, ev: Evidence) -> Dict[str, Any]:
        fv = ev.feature_vector or {}
        # Base features in [0,1]
        cosine = float(fv.get("cosine", 0.0))
        source_trust = float(fv.get("source_trust", 0.0))
        recency = float(fv.get("recency", 0.0))
        logic_ok = 1.0 if int(fv.get("logic_ok", 1)) else 0.0

        # Redundancy proxy: more supports -> higher redundancy [0,1]
        redundancy = 0.0
        if isinstance(ev.supports, list) and len(ev.supports) > 1:
            # Saturate at 1 when >= 5 supports
            redundancy = min(1.0, max(0.0, (len(ev.supports) - 1) / 4.0))

        # Majority polarity: '+' => +1, '-' => -1; tie defaults to +1
        pos = sum(1 for s in (ev.supports or []) if str(s.get("polarity", "+")) == "+")
        neg = sum(1 for s in (ev.supports or []) if str(s.get("polarity", "+")) == "-")
        sign = 1.0 if pos >= neg else -1.0

        # Weighted score in [0,1] before sign
        pos_weights = self.weights["cosine"] + self.weights["source_trust"] + self.weights["recency"] + self.weights["logic_ok"]
        # To keep bounded, normalize by sum of positive weights
        base = (
            cosine * self.weights["cosine"]
            + source_trust * self.weights["source_trust"]
            + recency * self.weights["recency"]
            + logic_ok * self.weights["logic_ok"]
        ) / max(1e-9, pos_weights)

        # Apply redundancy penalty (negative weight)
        base = max(0.0, min(1.0, base + redundancy * self.weights["redundancy_penalty"]))

        # Map to [-1, 1] by applying polarity directly to magnitude
        s_signed = sign * base

        delta = self.scale * s_signed

        return {
            "logit_delta": float(delta),
            "features": {
                "cosine": cosine,
                "source_trust": source_trust,
                "recency": recency,
                "logic_ok": logic_ok,
                "redundancy": redundancy,
                "polarity_sign": sign,
                "base": base,
                "signed": s_signed,
            },
        }


def update_belief(h: Hypothesis, delta_logit: float) -> Hypothesis:
    # Current belief
    b = float(h.belief if h.belief is not None else 0.5)
    # Clamp away from exact edges
    b = min(max(b, 0.01), 0.99)
    # Update in logit space
    new_logit = logit(b) + float(delta_logit)
    b_prime = sigmoid(new_logit)
    # Final clamp per spec
    b_prime = min(max(b_prime, 0.01), 0.99)

    # Heuristic CI shrink: width scaled by 1/(1 + |delta|), min width 0.05
    lo, hi = h.belief_ci if h.belief_ci else (0.0, 1.0)
    width = max(0.0, float(hi - lo)) or 0.5
    shrink = 1.0 / (1.0 + abs(float(delta_logit)))
    new_width = max(0.05, width * shrink)
    new_lo = max(0.0, b_prime - new_width / 2.0)
    new_hi = min(1.0, b_prime + new_width / 2.0)
    if new_lo > new_hi:
        new_lo, new_hi = new_hi, new_lo

    # Return updated instance (pydantic BaseModel is immutable by default unless configured; here we create a copy)
    return h.model_copy(update={"belief": b_prime, "belief_ci": (new_lo, new_hi)})


# Backward-compat wrapper to keep orchestrator usable
class Evaluator:
    def __init__(self) -> None:
        self.conf = ConfidenceEvaluator()

    def score(self, hypothesis: Hypothesis, evidences: List[Evidence]) -> float:
        if not evidences:
            return float(hypothesis.belief)
        # Combine deltas and apply
        total_delta = 0.0
        for ev in evidences:
            res = self.conf.score(ev)
            total_delta += float(res["logit_delta"]) / max(1, len(evidences))
        updated = update_belief(hypothesis, total_delta)
        return float(updated.belief)


