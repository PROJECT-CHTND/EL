# pyright: reportMissingImports=false, reportMissingModuleSource=false
import os
import sys
from typing import Any, Dict, List


# Ensure el-agent/src is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
EL_AGENT_SRC = os.path.join(REPO_ROOT, "el-agent", "src")
if EL_AGENT_SRC not in sys.path:
    sys.path.insert(0, EL_AGENT_SRC)

from el_agent.core.knowledge_integrator import KnowledgeIntegrator  # type: ignore[import]
from el_agent.schemas import Hypothesis  # type: ignore[import]


def _tokenize_simple(text: str) -> List[str]:
    import re

    return [t for t in re.split(r"[^\w\-]+", text.lower()) if t]


class ESIndexStub:
    def __init__(self) -> None:
        self._docs: List[Dict[str, Any]] = []

    def index(self, index: str, id: str, document: Dict[str, Any]) -> None:  # noqa: A003
        self._docs.append({"_id": str(id), "content": str(document.get("content", ""))})

    def search_bm25(self, index: str, query: str, size: int = 10) -> List[Dict[str, Any]]:
        q_tokens = set(_tokenize_simple(query))

        def score(doc: Dict[str, Any]) -> float:
            t_tokens = set(_tokenize_simple(doc.get("content", "")))
            overlap = len(q_tokens & t_tokens)
            length_bonus = 0.0001 * min(len(t_tokens), 500)
            return overlap + length_bonus

        ranked = sorted(self._docs, key=lambda d: (-score(d), str(d["_id"])))
        return ranked[:size]


class QdrantIndexStub:
    def __init__(self) -> None:
        self._points: List[Dict[str, Any]] = []

    def upsert(self, ids: List[str], payloads: List[Dict[str, Any]]) -> None:
        for pid, payload in zip(ids, payloads):
            self._points.append({"id": str(pid), "payload": {"text": str(payload.get("text", ""))}})

    def search(self, query_vector, top_k: int = 10) -> List[Dict[str, Any]]:  # noqa: ANN001
        # Integrator does not pass the query text, so we emulate a stable "semantic" ranking
        # using frequency of the token "猫" in payload.text to ensure predictable overlap.
        def score(pt: Dict[str, Any]) -> float:
            text = pt.get("payload", {}).get("text", "")
            return text.count("猫") + 0.0001 * len(text)

        ranked = sorted(self._points, key=lambda p: (-score(p), str(p["id"])))
        return ranked[:top_k]


def _make_jp_docs() -> List[Dict[str, str]]:
    # 10 Japanese documents with varying frequency of the token "猫"
    docs = [
        {"id": "d1", "text": "犬 は 友達。散歩 が 好き。"},
        {"id": "d2", "text": "猫 は 静か。家で 眠る。"},
        {"id": "d3", "text": "猫 の 行動 パターン は 夜行性。狩り を する。"},
        {"id": "d4", "text": "鳥 は 空 を 飛ぶ。"},
        {"id": "d5", "text": "猫 猫 猫 の 生態。猫 の 行動 は 多様。猫 は 学習 する。"},
        {"id": "d6", "text": "魚 は 泳ぐ。水中 生活。"},
        {"id": "d7", "text": "猫 と 犬 の 比較。性格 と 行動。"},
        {"id": "d8", "text": "人間 と 動物 の 共生。"},
        {"id": "d9", "text": "猫 は 高い 所 が 好き。落下 しても 平衡。"},
        {"id": "d10", "text": "昆虫 は 変態 する。"},
    ]
    return docs


def _index_docs(es: ESIndexStub, qd: QdrantIndexStub, docs: List[Dict[str, str]]) -> None:
    for d in docs:
        es.index(index="documents", id=d["id"], document={"content": d["text"]})
        qd.upsert(ids=[d["id"]], payloads=[{"text": d["text"]}])


def _normalize_doc_id(hit: Dict[str, Any]) -> str:
    return str(hit.get("_id") or hit.get("id") or hit.get("payload", {}).get("id"))


def test_rrf_prefers_overlap_with_japanese_docs():
    docs = _make_jp_docs()
    es = ESIndexStub()
    qd = QdrantIndexStub()
    _index_docs(es, qd, docs)

    ki = KnowledgeIntegrator(es_client=es, qdrant_client=qd)

    query = "猫 行動"
    ranked = ki.retrieve(query=query, k_bm25=10, k_vec=10, k_rrf=64)

    assert ranked, "retrieve() should return some documents"

    top_ids = [_normalize_doc_id(d) for d in ranked[:3]]
    # d5 contains most occurrences of "猫" and should appear in both lists and be favored by RRF
    assert "d5" in top_ids[:1], f"Expected overlap doc 'd5' to be top-ranked, got {top_ids[:3]}"


def test_sentence_extract_cap_japanese():
    docs = _make_jp_docs()
    es = ESIndexStub()
    qd = QdrantIndexStub()
    _index_docs(es, qd, docs)

    ki = KnowledgeIntegrator(es_client=es, qdrant_client=qd)

    query = "猫 行動"
    ranked = ki.retrieve(query=query, k_bm25=10, k_vec=10, k_rrf=64)
    hypothesis = Hypothesis(id="h_jp", text="猫 の 行動 パターン")
    sents = ki.sentence_extract(ranked, hypothesis)

    assert len(sents) <= 8
    assert len(sents) >= 1


