from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..schemas import Evidence, Hypothesis


def _tokenize(text: str) -> List[str]:
    import re

    if not text:
        return []
    # Split on non-word including Japanese punctuation
    tokens = re.split(r"[^\w\-]+", text.lower())
    return [t for t in tokens if t]


def _split_sentences(text: str) -> List[str]:
    import re

    if not text:
        return []
    # Basic rule-based splitter covering English and Japanese punctuation
    parts = re.split(r"(?<=[\.!?。！？])\s+|\n+", text)
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences


def _cosine_sim(a_tokens: Sequence[str], b_tokens: Sequence[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set, b_set = set(a_tokens), set(b_tokens)
    inter = len(a_set & b_set)
    if inter == 0:
        return 0.0
    import math

    return inter / math.sqrt(len(a_set) * len(b_set))


def _normalize_doc_id(hit: Dict[str, Any]) -> Optional[str]:
    # Try common keys across ES and Qdrant
    if "_id" in hit:
        return str(hit["_id"])
    if "id" in hit:
        return str(hit["id"])
    payload = hit.get("payload") if isinstance(hit, dict) else None
    if isinstance(payload, dict) and "id" in payload:
        return str(payload["id"])
    return None


def _rrf_merge(
    bm25_hits: List[Dict[str, Any]], vec_hits: List[Dict[str, Any]], k_rrf: int
) -> List[Tuple[str, float]]:
    # Compute rank maps
    bm25_ranks: Dict[str, int] = {}
    for rank, h in enumerate(bm25_hits, start=1):
        doc_id = _normalize_doc_id(h)
        if doc_id is not None and doc_id not in bm25_ranks:
            bm25_ranks[doc_id] = rank
    vec_ranks: Dict[str, int] = {}
    for rank, h in enumerate(vec_hits, start=1):
        doc_id = _normalize_doc_id(h)
        if doc_id is not None and doc_id not in vec_ranks:
            vec_ranks[doc_id] = rank

    # Candidate ids
    candidates = set(bm25_ranks) | set(vec_ranks)
    scores: List[Tuple[str, float]] = []
    for doc_id in candidates:
        s = 0.0
        if doc_id in bm25_ranks:
            s += 1.0 / (k_rrf + bm25_ranks[doc_id])
        if doc_id in vec_ranks:
            s += 1.0 / (k_rrf + vec_ranks[doc_id])
        scores.append((doc_id, s))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores


class KnowledgeIntegrator:
    def __init__(self, es_client: Any | None = None, qdrant_client: Any | None = None) -> None:
        self.es = es_client
        self.qdrant = qdrant_client

    def integrate(self, hypothesis: Hypothesis, evidences: List[Evidence]) -> Hypothesis:
        # Note: current Hypothesis model does not keep evidences; integration is a no-op placeholder.
        return hypothesis

    def retrieve(self, query: str, k_bm25: int = 200, k_vec: int = 200, k_rrf: int = 64) -> List[Dict[str, Any]]:
        # Stage 1: independent retrievals
        bm25_hits: List[Dict[str, Any]] = []
        if self.es is not None:
            try:
                bm25_hits = list(self.es.search_bm25(index="documents", query=query, size=k_bm25))
            except Exception:
                bm25_hits = []

        vec_hits: List[Dict[str, Any]] = []
        if self.qdrant is not None:
            try:
                # Stub: vectorization not implemented; tests will monkeypatch .search
                vec_hits = list(self.qdrant.search(query_vector=[0.0, 0.0, 0.0], top_k=k_vec))
            except Exception:
                vec_hits = []

        # Reciprocal Rank Fusion
        fused = _rrf_merge(bm25_hits, vec_hits, k_rrf=k_rrf)

        # Create lightweight doc entries for next stage
        id_to_doc: Dict[str, Dict[str, Any]] = {}
        for h in bm25_hits:
            doc_id = _normalize_doc_id(h)
            if doc_id is None:
                continue
            id_to_doc.setdefault(doc_id, {}).update(h)
        for h in vec_hits:
            doc_id = _normalize_doc_id(h)
            if doc_id is None:
                continue
            id_to_doc.setdefault(doc_id, {}).update(h)

        # Stage 2: cross-encoder re-rank (deterministic stub)
        q_tokens = _tokenize(query)

        def stub_score(doc: Dict[str, Any]) -> float:
            text = (
                doc.get("text")
                or doc.get("content")
                or doc.get("payload", {}).get("text")
                or ""
            )
            t_tokens = _tokenize(str(text))
            overlap = len(set(q_tokens) & set(t_tokens))
            length_bonus = 0.001 * min(len(t_tokens), 200)
            return overlap + length_bonus

        ranked_docs: List[Dict[str, Any]] = []
        for doc_id, rrf_score in fused:
            doc = id_to_doc.get(doc_id, {"id": doc_id})
            doc["rrf"] = rrf_score
            doc["rerank"] = stub_score(doc)
            ranked_docs.append(doc)

        ranked_docs.sort(key=lambda d: (-d["rerank"], -d.get("rrf", 0.0), str(_normalize_doc_id(d))))
        return ranked_docs[:16]

    def sentence_extract(self, docs: List[Dict[str, Any]], hypothesis: Hypothesis) -> List[str]:
        q_tokens = _tokenize(hypothesis.text)
        selected: List[str] = []
        for doc in docs:
            text = (
                doc.get("text")
                or doc.get("content")
                or doc.get("payload", {}).get("text")
                or ""
            )
            for sent in _split_sentences(str(text)):
                s_tokens = _tokenize(sent)
                if not s_tokens:
                    continue
                # keyword overlap
                if len(set(q_tokens) & set(s_tokens)) == 0:
                    continue
                # cosine threshold (stub)
                if _cosine_sim(q_tokens, s_tokens) < 0.6:
                    continue
                selected.append(sent)
                if len(selected) >= 8:
                    return selected
        return selected[:8]

    def to_evidence(self, sentences: List[str], hypothesis: Hypothesis) -> Evidence:
        supports = []
        q_tokens = _tokenize(hypothesis.text)
        cosines: List[float] = []
        for s in sentences:
            s_tokens = _tokenize(s)
            cos = _cosine_sim(q_tokens, s_tokens)
            cosines.append(cos)
            supports.append(
                {
                    "hypothesis": hypothesis.id,
                    "polarity": "+",
                    "span": s,
                    "score_raw": cos,
                    "source": "stub",
                    "time": 0,
                }
            )
        avg_cos = sum(cosines) / len(cosines) if cosines else 0.0
        feature_vector = {
            "cosine": avg_cos,
            "source_trust": 0.5,
            "recency": 0.5,
            "logic_ok": 1,
        }
        return Evidence(entities=[], relations=[], supports=supports, feature_vector=feature_vector)


