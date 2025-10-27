from __future__ import annotations

import math
import time
from typing import List, Tuple

from agent.models.kg import KGPayload
from agent.slots import Slot, SlotRegistry

DEFAULT_IMPORTANCE = 0.5
RECENCY_DECAY_SECS = 60 * 60 * 24 * 7  # 1 week half-life


def _staleness(last_ts: float | None) -> float:
    """Compute staleness boost based on last filled timestamp (exponential)."""

    if last_ts is None:
        return 1.0
    age = max(0.0, time.time() - last_ts)
    if age <= 0:
        return 1.0
    return 1.0 - math.exp(-age / RECENCY_DECAY_SECS)


def analyze_gaps(registry: SlotRegistry, kg: KGPayload) -> List[Tuple[Slot, float]]:
    """Return slots sorted by priority descending."""

    results: List[Tuple[Slot, float]] = []
    for slot in registry.all_slots():
        importance = slot.importance or DEFAULT_IMPORTANCE
        filled_ratio = max(0.0, min(1.0, slot.filled_ratio))
        staleness = _staleness(slot.last_filled_ts)
        priority = importance * (1 - filled_ratio) * staleness
        results.append((slot, priority))

    results.sort(key=lambda pair: pair[1], reverse=True)
    return results
