from __future__ import annotations

from typing import List, Optional

from agent.kg.client import Neo4jClient
from agent.models.kg import KGPayload

DEFAULT_STAGE_WEIGHT = 1.0


def calculate_confidence(logprobs: List[float], stage_weight: float = DEFAULT_STAGE_WEIGHT) -> float:
    """Calculate confidence score based on mean logprob and stage weight."""
    if not logprobs:
        return 0.5 * stage_weight  # fallback default
    mean_lp = sum(logprobs) / len(logprobs)
    # Map logprob (usually negative) → (0,1] via exponential
    prob = pow(2.718281828, mean_lp)
    return max(0.0, min(1.0, prob * stage_weight))


def merge_and_persist(
    payload: KGPayload,
    *,
    logprobs: Optional[List[float]] = None,
    stage_weight: float = DEFAULT_STAGE_WEIGHT,
    neo4j_client: Optional[Neo4jClient] = None,
) -> KGPayload:
    """Merge KG fragment into Neo4j with confidence estimation.

    1. Estimate confidence using provided logprobs and stage weight.
    2. Update entity & relation confidence fields.
    3. Persist via Neo4j MERGE semantics.
    """

    confidence = calculate_confidence(logprobs or [], stage_weight=stage_weight)

    for ent in payload.entities:
        ent.confidence = confidence
    for rel in payload.relations:
        rel.confidence = confidence

    client = neo4j_client or Neo4jClient()
    client.ingest(payload)
    return payload 