from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import logging
import os
from pathlib import Path

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
        self._logger = logging.getLogger(__name__)

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

        # Optional calibrated logistic-regression weights
        self._calibrated: Optional[Dict[str, Any]] = self._load_calibrated_weights()
        if self._calibrated is None:
            self._logger.info("ConfidenceEvaluator: using default heuristic weights")
        else:
            src = self._calibrated.get("_source_path", "<unknown>")
            self._logger.info("ConfidenceEvaluator: loaded calibrated weights from %s", src)

    def _load_calibrated_weights(self) -> Optional[Dict[str, Any]]:
        # 1) Environment variable takes precedence
        env_path = os.getenv("EL_EVAL_WEIGHTS")
        candidate_paths: List[Path] = []
        if env_path:
            candidate_paths.append(Path(env_path).expanduser())

        # 2) Search upward for config/weights/weights.json from this file location
        try:
            here = Path(__file__).resolve()
            for parent in [here] + list(here.parents):
                candidate = parent / "config" / "weights" / "weights.json"
                if candidate.exists():
                    candidate_paths.append(candidate)
        except Exception:
            pass

        # Deduplicate while preserving order
        seen = set()
        unique_paths: List[Path] = []
        for p in candidate_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        for path in unique_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Validate schema: {"intercept": float, "coef": {str: float}}
                if not isinstance(data, dict):
                    continue
                if "intercept" not in data or "coef" not in data:
                    continue
                coef = data.get("coef", {})
                if not isinstance(coef, dict):
                    continue
                # Cast to floats
                intercept = float(data.get("intercept", 0.0))
                coef_cast: Dict[str, float] = {}
                for k, v in coef.items():
                    try:
                        coef_cast[str(k)] = float(v)
                    except Exception:
                        continue
                out = {"intercept": intercept, "coef": coef_cast, "_source_path": str(path)}
                return out
            except Exception:
                continue
        return None

    def score(self, ev: Evidence) -> Dict[str, Any]:
        fv = ev.feature_vector or {}
        # Base features in [0,1]
        cosine = float(fv.get("cosine", 0.0))
        source_trust = float(fv.get("source_trust", 0.0))
        recency = float(fv.get("recency", 0.0))
        logic_ok = 1.0 if int(fv.get("logic_ok", 1)) else 0.0

        # Redundancy proxy: more supports -> higher redundancy [0,1]
        redundancy = float(fv.get("redundancy", 0.0))
        if "redundancy" not in fv:
            if isinstance(ev.supports, list) and len(ev.supports) > 1:
                redundancy = min(1.0, max(0.0, (len(ev.supports) - 1) / 4.0))
            else:
                redundancy = 0.0

        # Majority polarity: '+' => +1, '-' => -1; tie defaults to +1
        pos = sum(1 for s in (ev.supports or []) if str(s.get("polarity", "+")) == "+")
        neg = sum(1 for s in (ev.supports or []) if str(s.get("polarity", "+")) == "-")
        sign = 1.0 if pos >= neg else -1.0

        # If calibrated LR weights are available, use them to compute logit_delta directly
        if self._calibrated is not None:
            coef_map: Dict[str, float] = self._calibrated.get("coef", {})  # type: ignore[assignment]
            intercept: float = float(self._calibrated.get("intercept", 0.0))  # type: ignore[assignment]

            feature_values: Dict[str, float] = {
                "cosine": cosine,
                "source_trust": source_trust,
                "recency": recency,
                "logic_ok": logic_ok,
                "redundancy": redundancy,
                "polarity_sign": sign,
            }
            # Only multiply features present in coef_map
            linear_sum = intercept
            used: Dict[str, float] = {}
            for name, w in coef_map.items():
                val = float(feature_values.get(name, 0.0))
                linear_sum += float(w) * val
                used[name] = val

            return {
                "logit_delta": float(linear_sum),
                "features": {
                    **feature_values,
                    "used_in_model": used,
                },
            }

        # Fallback heuristic path
        pos_weights = (
            self.weights["cosine"] + self.weights["source_trust"] + self.weights["recency"] + self.weights["logic_ok"]
        )
        base = (
            cosine * self.weights["cosine"]
            + source_trust * self.weights["source_trust"]
            + recency * self.weights["recency"]
            + logic_ok * self.weights["logic_ok"]
        ) / max(1e-9, pos_weights)
        base = max(0.0, min(1.0, base + redundancy * self.weights["redundancy_penalty"]))
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


