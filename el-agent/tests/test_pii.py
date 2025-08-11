import json
import types
import os
from httpx import AsyncClient, ASGITransport
import pytest

from el_agent.app import app  # type: ignore
from el_agent.utils.pii import mask_text, mask_structure, user_id_hash  # type: ignore


class _StubRedis:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value


@pytest.mark.asyncio
async def test_masking_in_markdown_and_wal(tmp_path, monkeypatch):
    # Ensure salt is set for deterministic hashes in test
    monkeypatch.setenv("PII_SALT", "test_salt")

    # Stub redis module used in app
    import el_agent.app as app_mod  # type: ignore

    stub = _StubRedis()
    fake_redis_mod = types.SimpleNamespace(from_url=lambda *args, **kwargs: stub)
    monkeypatch.setattr(app_mod, "redis", fake_redis_mod, raising=False)

    # Prepare a payload containing PII
    payload = {
        "user_msg": "Contact me at john.doe@example.com or +1 (212) 555-1234. I live at 123 Main St.",
        "session_id": "sess-abc",
    }

    # Call API
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/respond", json=payload)
        assert res.status_code == 200
        data = res.json()

    # Markdown should be masked
    md = data.get("markdown", "")
    assert "example.com" not in md
    assert "555-1234" not in md
    assert "Main St" not in md
    # tokens should appear
    assert "<email:" in md
    assert "<phone:" in md or "<address:" in md

    # Check WAL latest line for user_id_hash; open today's log
    from el_agent.core.orchestrator import _wal_dir  # type: ignore

    wal_dir = _wal_dir()
    # find today's log file
    found = False
    if wal_dir.exists():
        for p in sorted(wal_dir.glob("*.log"), reverse=True):
            with p.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if not lines:
                continue
            last = json.loads(lines[-1])
            # Expect masking and user_id_hash
            assert "user_id_hash" in last
            # Ensure data has masked tokens when strings present
            blob = json.dumps(last.get("data", {}), ensure_ascii=False)
            assert "<email:" in blob or "<phone:" in blob or "<address:" in blob or blob == "{}"
            found = True
            break
    assert found, "WAL log file was not found or had no entries"


def test_mask_text_basic(monkeypatch):
    monkeypatch.setenv("PII_SALT", "abc")
    s = "Email a@b.co, phone +81 90-1234-5678, addr 10 Downing St"
    out = mask_text(s)
    assert "a@b.co" not in out
    assert "1234-5678" not in out
    assert "Downing St" not in out
    assert "<email:" in out and "<phone:" in out and "<address:" in out


def test_mask_structure_recursive(monkeypatch):
    monkeypatch.setenv("PII_SALT", "abc")
    payload = {
        "a": "john@x.com",
        "b": ["+1-202-000-0000", {"c": "77 Sunset Blvd"}],
    }
    masked = mask_structure(payload)
    as_text = json.dumps(masked)
    assert "john@x.com" not in as_text
    assert "+1-202-000-0000" not in as_text
    assert "Sunset Blvd" not in as_text
    assert "<email:" in as_text and "<phone:" in as_text and "<address:" in as_text


def test_user_id_hash(monkeypatch):
    monkeypatch.setenv("PII_SALT", "xyz")
    h = user_id_hash("session-1")
    assert isinstance(h, str) and len(h) == 16
