"""
EL SDK - Python Client for Eager Learner API

Usage:
    from el_sdk import ELClient

    client = ELClient(base_url="http://localhost:8000", api_key="your-key")
    session = client.create_session(topic="引き継ぎ")
    question = session.get_next_question()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class Fact:
    id: str
    content: str
    category: str
    tags: list[str]
    importance: float = 0.0


@dataclass
class Question:
    id: str
    text: str
    type: str
    priority: str
    context: dict = field(default_factory=dict)


@dataclass
class SessionSummary:
    total_facts: int
    coverage_score: float
    key_findings: list[str]
    remaining_gaps: list[str]
    inconsistencies: list[dict]


class Session:
    """ELインタビューセッション"""

    def __init__(self, session_id: str, client: ELClient):
        self.id = session_id
        self._client = client

    def upload_document(self, file_path: str, description: str = "") -> dict:
        """ドキュメントをアップロードしてファクト抽出を開始"""
        with open(file_path, "rb") as f:
            return self._client._request(
                "POST",
                f"/sessions/{self.id}/documents",
                files={"file": f},
                data={"description": description},
            )

    def get_next_question(self) -> Optional[Question]:
        """次の質問を取得"""
        resp = self._client._request("GET", f"/sessions/{self.id}/questions/next")
        if resp is None:
            return None
        return Question(
            id=resp["id"],
            text=resp["text"],
            type=resp["type"],
            priority=resp["priority"],
            context=resp.get("context", {}),
        )

    def respond(self, question_id: str, response_text: str, confidence: str = "medium") -> dict:
        """質問に回答"""
        return self._client._request(
            "POST",
            f"/sessions/{self.id}/responses",
            json={
                "question_id": question_id,
                "response_text": response_text,
                "confidence": confidence,
            },
        )

    def skip_question(self, question_id: str) -> dict:
        """質問をスキップ"""
        return self._client._request(
            "POST", f"/sessions/{self.id}/questions/{question_id}/skip"
        )

    def get_facts(self, category: str = None, tag: str = None) -> list[Fact]:
        """蓄積されたファクトを取得"""
        params = {}
        if category:
            params["category"] = category
        if tag:
            params["tag"] = tag
        resp = self._client._request("GET", f"/sessions/{self.id}/facts", params=params)
        return [
            Fact(
                id=f["id"],
                content=f["content"],
                category=f["category"],
                tags=f.get("tags", []),
                importance=f.get("importance", 0.0),
            )
            for f in resp.get("facts", [])
        ]

    def get_summary(self) -> SessionSummary:
        """ナレッジサマリーを取得"""
        resp = self._client._request("GET", f"/sessions/{self.id}/summary")
        return SessionSummary(
            total_facts=resp["total_facts"],
            coverage_score=resp["coverage_score"],
            key_findings=resp["key_findings"],
            remaining_gaps=resp["remaining_gaps"],
            inconsistencies=resp.get("inconsistencies", []),
        )

    def get_knowledge_map(self) -> dict:
        """ナレッジマップデータを取得"""
        return self._client._request("GET", f"/sessions/{self.id}/knowledge-map")


class ELClient:
    """EL APIクライアント"""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        timeout: int = 30,
    ):
        self.base_url = (base_url or os.getenv("EL_API_URL", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.getenv("EL_LICENSE_KEY", "")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}/api/v1{path}"

        if "files" in kwargs:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
        else:
            resp = self._session.request(method, url, timeout=self.timeout, **kwargs)

        resp.raise_for_status()

        if resp.status_code == 204:
            return {}
        return resp.json()

    def create_session(
        self,
        topic: str,
        description: str = "",
        tags: list[str] = None,
    ) -> Session:
        """新しいインタビューセッションを作成"""
        resp = self._request(
            "POST",
            "/sessions",
            json={
                "topic": topic,
                "description": description,
                "tags": tags or [],
            },
        )
        return Session(session_id=resp["id"], client=self)

    def get_session(self, session_id: str) -> Session:
        """既存のセッションに接続"""
        self._request("GET", f"/sessions/{session_id}")
        return Session(session_id=session_id, client=self)

    def list_sessions(self, status: str = "all", limit: int = 20) -> list[dict]:
        """セッション一覧を取得"""
        return self._request(
            "GET", "/sessions", params={"status": status, "limit": limit}
        )

    def health_check(self) -> dict:
        """APIの稼働状態を確認"""
        return self._request("GET", "/health")
