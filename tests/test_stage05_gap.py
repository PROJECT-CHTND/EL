import time

from prometheus_client import REGISTRY

from agent.models.kg import KGPayload
from agent.pipeline.stage05_gap import analyze_gaps
from agent.slots import Slot, SlotRegistry
from agent.monitoring.metrics import SLOT_GAP_LATENCY_SECONDS


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
    try:
        SLOT_GAP_LATENCY_SECONDS.remove("stage05_gap")
    except KeyError:
        pass

    ranked = analyze_gaps(registry, kg)

    # The unfilled high importance slot should come first
    assert ranked[0][0].name == "slot1"

    # Filled slot should have zero priority
    filled_priority = next(p for s, p in ranked if s.name == "slot2")
    assert filled_priority == 0.0

    latency_value = REGISTRY.get_sample_value("slot_gap_latency_seconds", {"pipeline_stage": "stage05_gap"})
    assert latency_value is not None
    assert latency_value >= 0.0