from __future__ import annotations

from typing import Any, Dict, List

import pytest

from el_agent.core.knowledge_integrator import KnowledgeIntegrator
from el_agent.schemas import Hypothesis


class _ESStub:
    def __init__(self, hits: List[Dict[str, Any]]):
        self._hits = hits

    def search_bm25(self, index: str, query: str, size: int = 10):
        return self._hits[:size]


class _QdrantStub:
    def __init__(self, hits: List[Dict[str, Any]]):
        self._hits = hits

    def search(self, query_vector, top_k: int = 10):
        return self._hits[:top_k]


def _doc(doc_id: str, text: str) -> Dict[str, Any]:
    return {"_id": doc_id, "text": text}


def test_rrf_favors_overlap():
    # d2 appears in both lists; should rank above non-overlap with similar content
    es_hits = [_doc("d1", "alpha beta"), _doc("d2", "alpha gamma"), _doc("d3", "delta")]
    qd_hits = [
        {"id": "d2", "payload": {"text": "alpha gamma"}},
        {"id": "d4", "payload": {"text": "epsilon"}},
    ]
    ki = KnowledgeIntegrator(es_client=_ESStub(es_hits), qdrant_client=_QdrantStub(qd_hits))
    ranked = ki.retrieve(query="alpha", k_bm25=3, k_vec=2, k_rrf=64)
    top_ids = [str(d.get("_id") or d.get("id")) for d in ranked[:2]]
    assert top_ids[0] in {"d2"}


def test_sentence_extract_cap():
    ki = KnowledgeIntegrator()
    hypothesis = Hypothesis(id="h1", text="alpha beta gamma")
    docs = [{"_id": f"d{i}", "text": ("alpha beta. gamma alpha. " * 10)} for i in range(2)]
    sents = ki.sentence_extract(docs, hypothesis)
    assert len(sents) <= 8
    # should return at least some sentences, but capped
    assert len(sents) >= 1


