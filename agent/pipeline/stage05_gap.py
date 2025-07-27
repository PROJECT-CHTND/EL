from __future__ import annotations

import time
from typing import List, Tuple

from agent.models.kg import KGPayload
from agent.slots import Slot, SlotRegistry

DEFAULT_IMPORTANCE = 0.5
RECENCY_DECAY_SECS = 60 * 60 * 24 * 7  # 1 week half-life


def _recency_boost(last_ts: float | None) -> float:
    """Compute recency boost based on last filled timestamp (exponential decay)."""
    if last_ts is None:
        return 1.0
    age = time.time() - last_ts
    return max(0.1, 0.5 ** (age / RECENCY_DECAY_SECS))


def analyze_gaps(registry: SlotRegistry, kg: KGPayload) -> List[Tuple[Slot, float]]:
    """Return slots sorted by priority descending.

    priority = importance × (1 - filled_ratio) × recency_boost
    For current simplistic approach, filled_ratio is 1 if slot.filled else 0.
    """

    results: List[Tuple[Slot, float]] = []
    for slot in registry.all_slots():
        importance = slot.importance or DEFAULT_IMPORTANCE
        filled_ratio = 1.0 if slot.filled else 0.0
        recency = _recency_boost(slot.last_filled_ts)
        priority = importance * (1 - filled_ratio) * recency
        results.append((slot, priority))

    # sort by priority descending
    results.sort(key=lambda p: p[1], reverse=True)
    return results 