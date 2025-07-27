from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class Slot(BaseModel):
    """Representation of a knowledge slot to be filled."""

    name: str = Field(..., description="Slot identifier")
    description: str = Field(..., description="Human-readable description of the slot")
    type: str | None = Field(None, description="Slot type/category")
    importance: float = Field(1.0, ge=0.0, le=1.0, description="Intrinsic importance weight 0-1")
    filled: bool = Field(False, description="Whether the slot is already filled")
    last_filled_ts: float | None = Field(None, description="Unix timestamp when slot was filled (epoch seconds)")


class SlotRegistry:
    """Simple in-memory registry for slots."""

    def __init__(self) -> None:
        self._slots: Dict[str, Slot] = {}

    def add(self, slot: Slot) -> None:
        self._slots[slot.name] = slot

    def mark_filled(self, name: str) -> None:
        if name in self._slots:
            self._slots[name].filled = True

    def unfilled_slots(self) -> List[Slot]:
        return [s for s in self._slots.values() if not s.filled]

    def all_slots(self) -> List[Slot]:
        return list(self._slots.values()) 