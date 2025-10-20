from __future__ import annotations

import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

FILL_THRESHOLD = 0.85


class Slot(BaseModel):
    """Representation of a knowledge slot to be filled."""

    name: str = Field(..., description="Slot identifier")
    description: str = Field(..., description="Human-readable description of the slot")
    type: str | None = Field(None, description="Slot type/category")
    importance: float = Field(1.0, ge=0.0, le=1.0, description="Intrinsic importance weight 0-1")
    filled_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Degree of completion confidence for the slot"
    )
    last_filled_ts: float | None = Field(
        None, description="Unix timestamp when slot was filled (epoch seconds)"
    )
    evidence_ids: List[str] = Field(default_factory=list, description="Associated evidence IDs")
    source_kind: str | None = Field(
        None, description="Origin of the evidence filling this slot (user/retrieval/model)"
    )
    value: str | None = Field(None, description="Most recent value captured for the slot")

    @property
    def filled(self) -> bool:
        """Whether the slot is considered filled based on filled_ratio."""

        return self.filled_ratio >= FILL_THRESHOLD


class SlotRegistry:
    """Simple in-memory registry for slots."""

    def __init__(self) -> None:
        self._slots: Dict[str, Slot] = {}

    def add(self, slot: Slot) -> None:
        self._slots[slot.name] = slot

    def get(self, name: str) -> Slot | None:
        return self._slots.get(name)

    def update(
        self,
        name: str,
        *,
        value: Optional[str] = None,
        filled_ratio: float = 1.0,
        evidence_ids: Optional[List[str]] = None,
        source_kind: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> Slot | None:
        """Update a slot with new information and return the updated slot."""

        slot = self._slots.get(name)
        if slot is None:
            return None

        ts = timestamp if timestamp is not None else time.time()
        new_ratio = max(slot.filled_ratio, min(1.0, filled_ratio))
        update_payload = {
            "filled_ratio": new_ratio,
            "last_filled_ts": ts if new_ratio > slot.filled_ratio else slot.last_filled_ts,
            "value": value if value is not None else slot.value,
            "source_kind": source_kind or slot.source_kind,
            "evidence_ids": evidence_ids if evidence_ids is not None else slot.evidence_ids,
        }
        updated = slot.model_copy(update=update_payload)
        self._slots[name] = updated
        return updated

    def unfilled_slots(self) -> List[Slot]:
        return [s for s in self._slots.values() if not s.filled]

    def all_slots(self) -> List[Slot]:
        return list(self._slots.values())

    def is_all_filled(self) -> bool:
        return bool(self._slots) and all(slot.filled for slot in self._slots.values())

    def coverage(self) -> float:
        if not self._slots:
            return 0.0
        return sum(1 for slot in self._slots.values() if slot.filled) / len(self._slots)
