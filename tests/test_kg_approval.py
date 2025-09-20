from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from agent.api.main import app
from agent.security.jwt_auth import verify_jwt


class DummyClient:
    def __init__(self) -> None:
        self.storage: dict[str, dict[str, Any]] = {}

    def submit_fact(self, fact):
        fid = fact.id or "f-1"
        status = "pending" if fact.impact == "critical" else "confirmed"
        obj = {"id": fid, "text": fact.text, "belief": fact.belief, "impact": fact.impact, "status": status}
        self.storage[fid] = obj
        return obj

    def approve_fact(self, fact_id: str, decision: str, approver: str | None = None):
        obj = self.storage.get(fact_id)
        if not obj:
            raise ValueError("not found")
        obj["status"] = "confirmed" if decision == "approve" else "rejected"
        return obj


def override_verify_jwt():  # type: ignore
    return {"sub": "tester"}


def test_critical_fact_requires_approval(monkeypatch):
    # monkeypatch the client used in API module
    from agent.api import main as api_main
    dummy = DummyClient()
    monkeypatch.setattr(api_main, "Neo4jClient", lambda: dummy)

    # override JWT dependency
    app.dependency_overrides[verify_jwt] = override_verify_jwt  # type: ignore

    c = TestClient(app)

    # Submit critical
    res = c.post(
        "/kg/submit",
        headers={"Authorization": "Bearer test"},
        json={"text": "X is critical", "belief": 0.9, "impact": "critical"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "pending"

    fid = body["id"]
    # Approve
    res2 = c.post(
        "/kg/approve",
        headers={"Authorization": "Bearer test"},
        json={"fact_id": fid, "decision": "approve"},
    )
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2["status"] == "confirmed"


def test_normal_fact_auto_confirms(monkeypatch):
    from agent.api import main as api_main
    dummy = DummyClient()
    monkeypatch.setattr(api_main, "Neo4jClient", lambda: dummy)

    app.dependency_overrides[verify_jwt] = override_verify_jwt  # type: ignore

    c = TestClient(app)
    res = c.post(
        "/kg/submit",
        headers={"Authorization": "Bearer test"},
        json={"text": "X is normal", "belief": 0.7, "impact": "normal"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "confirmed"


def test_reject_transitions_to_rejected(monkeypatch):
    from agent.api import main as api_main
    dummy = DummyClient()
    monkeypatch.setattr(api_main, "Neo4jClient", lambda: dummy)

    app.dependency_overrides[verify_jwt] = override_verify_jwt  # type: ignore

    c = TestClient(app)
    res = c.post(
        "/kg/submit",
        headers={"Authorization": "Bearer test"},
        json={"text": "X is critical", "belief": 0.9, "impact": "critical"},
    )
    fid = res.json()["id"]

    res2 = c.post(
        "/kg/approve",
        headers={"Authorization": "Bearer test"},
        json={"fact_id": fid, "decision": "reject"},
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "rejected"


