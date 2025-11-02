import time

from agent.models.kg import KGPayload
from agent.pipeline.stage05_gap import analyze_gaps
from agent.slots import Slot, SlotRegistry


def test_analyze_gaps_priority_order():
    registry = SlotRegistry()
    registry.add(Slot(name="slot1", description="desc1", importance=0.9))
    registry.add(Slot(name="slot2", description="desc2", importance=1.0, filled_ratio=1.0))
    registry.add(
        Slot(
            name="slot3",
            description="desc3",
            importance=0.5,
            last_filled_ts=time.time() - 60 * 60 * 24 * 10,  # 10 days ago
        )
    )

    kg = KGPayload(entities=[], relations=[])
    ranked = analyze_gaps(registry, kg)

    # The unfilled high importance slot should come first
    assert ranked[0][0].name == "slot1"

    # Filled slot should have zero priority
    filled_priority = next(p for s, p in ranked if s.name == "slot2")
    assert filled_priority == 0.0 