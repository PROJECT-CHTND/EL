import time

import pytest

from agent.slots import Slot, SlotRegistry
from agent.stores.redis_slot_registry import (
    InMemorySlotRegistryRepository,
    RedisSlotRegistryRepository,
    SlotRegistryPersistenceManager,
)
from nani_bot import ThinkingSession


class _StubRedis:
    def __init__(self):
        self._store: dict[str, str] = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str):
        self._store[key] = value

    async def delete(self, key: str):
        self._store.pop(key, None)


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


@pytest.mark.asyncio
async def test_redis_slot_repository_roundtrip():
    stub = _StubRedis()
    repo = RedisSlotRegistryRepository(stub)

    registry = SlotRegistry()
    registry.add(Slot(name="summary", description="desc"))
    registry.update(
        "summary",
        value="captured",
        filled_ratio=0.9,
        evidence_ids=["e1"],
        source_kind="user",
        timestamp=123.0,
    )

    await repo.save("session-1", registry)

    restored = await repo.load("session-1")
    assert restored is not None
    slot = restored.get("summary")
    assert slot is not None
    assert slot.value == "captured"
    assert slot.evidence_ids == ["e1"]


@pytest.mark.asyncio
async def test_slot_registry_persists_across_session_restart():
    stub = _StubRedis()
    manager = SlotRegistryPersistenceManager(
        primary=RedisSlotRegistryRepository(stub),
        fallback=InMemorySlotRegistryRepository(),
    )

    first_session = ThinkingSession("user", "障害の振り返り", 777, "Japanese")
    first_session.configure_slots("postmortem")
    first_session.slot_registry.update("summary", value="一次対応を実施", filled_ratio=1.0)
    await manager.save(first_session.slot_registry_storage_id, first_session.slot_registry)

    restarted = ThinkingSession("user", "障害の振り返り", 777, "Japanese")
    restarted.configure_slots("postmortem")
    restored_registry = await manager.load(restarted.slot_registry_storage_id)
    assert restored_registry is not None
    restarted.slot_registry = restored_registry

    summary_slot = restarted.slot_registry.get("summary")
    assert summary_slot is not None
    assert summary_slot.value == "一次対応を実施"
