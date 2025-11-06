from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Protocol

from agent.slots import SlotRegistry


class SlotRegistryRepository(Protocol):
    """Minimal protocol for slot-registry persistence backends."""

    async def load(self, session_id: str) -> Optional[SlotRegistry]:
        ...

    async def save(self, session_id: str, registry: SlotRegistry) -> None:
        ...

    async def delete(self, session_id: str) -> None:
        ...


class RedisSlotRegistryRepository:
    """Persist :class:`SlotRegistry` objects to Redis as JSON blobs."""

    def __init__(self, client: Any, *, prefix: str = "slot-registry") -> None:
        if client is None:
            raise ValueError("Redis client must not be None")
        self._client = client
        self._prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}:{session_id}"

    async def load(self, session_id: str) -> Optional[SlotRegistry]:
        raw = await self._client.get(self._key(session_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data: Mapping[str, Any] = json.loads(raw)
        return SlotRegistry.model_validate(data)

    async def save(self, session_id: str, registry: SlotRegistry) -> None:
        payload = json.dumps(registry.model_dump(), ensure_ascii=False)
        await self._client.set(self._key(session_id), payload)

    async def delete(self, session_id: str) -> None:
        await self._client.delete(self._key(session_id))


class InMemorySlotRegistryRepository:
    """Simple in-memory fallback repository for slot registries."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    async def load(self, session_id: str) -> Optional[SlotRegistry]:
        data = self._store.get(session_id)
        if data is None:
            return None
        return SlotRegistry.model_validate(data)

    async def save(self, session_id: str, registry: SlotRegistry) -> None:
        self._store[session_id] = registry.model_dump()

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class SlotRegistryPersistenceManager:
    """Coordinate persistence between primary and fallback repositories."""

    def __init__(
        self,
        *,
        primary: SlotRegistryRepository | None = None,
        fallback: SlotRegistryRepository | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback or InMemorySlotRegistryRepository()

    async def load(self, session_id: str) -> Optional[SlotRegistry]:
        if self._primary is not None:
            try:
                registry = await self._primary.load(session_id)
            except Exception:
                self._primary = None
            else:
                if registry is not None:
                    return registry
        if self._fallback is None:
            return None
        return await self._fallback.load(session_id)

    async def save(self, session_id: str, registry: SlotRegistry) -> None:
        stored = False
        if self._primary is not None:
            try:
                await self._primary.save(session_id, registry)
                stored = True
            except Exception:
                self._primary = None
        if not stored and self._fallback is not None:
            await self._fallback.save(session_id, registry)

    async def delete(self, session_id: str) -> None:
        deleted = False
        if self._primary is not None:
            try:
                await self._primary.delete(session_id)
                deleted = True
            except Exception:
                self._primary = None
        if not deleted and self._fallback is not None:
            await self._fallback.delete(session_id)
