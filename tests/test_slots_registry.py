import time

from agent.slots import Slot, SlotRegistry


def test_slot_registry_update_merges_evidence_and_updates_timestamp():
    registry = SlotRegistry()
    registry.add(Slot(name="summary", description="desc"))

    base_ts = time.time()
    first = registry.update(
        "summary",
        value="first",
        filled_ratio=0.6,
        evidence_ids=["e1"],
        source_kind="user",
        timestamp=base_ts,
    )
    assert first is not None
    assert first.filled_ratio == 0.6
    assert first.last_filled_ts == base_ts
    assert first.evidence_ids == ["e1"]

    later_ts = base_ts + 5
    second = registry.update(
        "summary",
        value="second",
        filled_ratio=0.4,
        evidence_ids=["e1", "e2"],
        source_kind="user",
        timestamp=later_ts,
    )
    assert second is not None
    # filled_ratio should not decrease below the previous maximum
    assert second.filled_ratio == 0.6
    # timestamp refreshed because value changed even with same ratio
    assert second.last_filled_ts == later_ts
    assert second.value == "second"
    assert second.evidence_ids == ["e1", "e2"]


def test_slot_registry_update_refreshes_timestamp_on_new_evidence():
    registry = SlotRegistry()
    registry.add(Slot(name="impact", description="desc"))

    base = registry.update("impact", value="first", filled_ratio=0.3, evidence_ids=["a"], timestamp=100.0)
    assert base is not None
    assert base.last_filled_ts == 100.0

    updated = registry.update("impact", filled_ratio=0.3, evidence_ids=["a", "b"], timestamp=200.0)
    assert updated is not None
    assert updated.evidence_ids == ["a", "b"]
    # timestamp refreshed because new evidence arrived
    assert updated.last_filled_ts == 200.0
