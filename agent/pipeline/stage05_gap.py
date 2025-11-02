from __future__ import annotations

import math
import time
from typing import List, Tuple

from agent.models.kg import KGPayload
from agent.slots import Slot, SlotRegistry

DEFAULT_IMPORTANCE = 0.5
# τ: 時定数（約7日）
TAU_SECS = 60 * 60 * 24 * 7


def _staleness(last_ts: float | None, now: float | None = None) -> float:
    """staleness = 1 − exp(−Δt/τ)."""
    if last_ts is None:
        return 1.0
    t_now = now or time.time()
    delta = max(0.0, t_now - last_ts)
    return 1.0 - math.exp(-delta / TAU_SECS)


def analyze_gaps(registry: SlotRegistry, kg: KGPayload) -> List[Tuple[Slot, float]]:
    """Return slots sorted by canonical priority.

    priority = importance × (1 − filled_ratio) × staleness
    For the current implementation, filled_ratio = 1 if slot.filled else 0.
    """

    results: List[Tuple[Slot, float]] = []
    now = time.time()
    for slot in registry.all_slots():
        importance = slot.importance or DEFAULT_IMPORTANCE
        filled_ratio = 1.0 if slot.filled else 0.0
        st = _staleness(slot.last_filled_ts, now)
        priority = importance * (1 - filled_ratio) * st
        results.append((slot, priority))

    results.sort(key=lambda p: p[1], reverse=True)
    return results


def top_k_slots(registry: SlotRegistry, kg: KGPayload, k: int = 3, temperature: float = 1.0) -> List[Tuple[Slot, float]]:
    """Return top-k slots with optional softmax weighting for analysis.

    If temperature != 1.0, priorities are transformed p' = softmax(p / T).
    The return list remains sorted by raw priority, with weights encoded in the score.
    """
    pairs = analyze_gaps(registry, kg)
    top = pairs[: max(1, k)]
    if not top:
        return []
    if temperature <= 0:
        return top

    # Compute softmax weights (informational; we return weighted scores)
    import math as _m

    vals = [s for _, s in top]
    max_v = max(vals)
    exps = [_m.exp((v - max_v) / max(1e-6, temperature)) for v in vals]
    z = sum(exps) or 1.0
    weights = [e / z for e in exps]

    weighted = [(slot, float(w)) for (slot, _), w in zip(top, weights)]
    return weighted