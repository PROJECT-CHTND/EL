from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    SearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ProductQuantization,
    ProductQuantizationConfig,
)


class QdrantStore:
    def __init__(self, url: str, collection: str, vector_size: int):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        collections = {c.name for c in self.client.get_collections().collections}
        if self.collection in collections:
            return
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=10000),
        )

    def upsert_vectors(self, ids: Sequence[int], vectors: Sequence[Sequence[float]], payloads: Optional[Sequence[dict]] = None) -> None:
        # Use batch upsert to satisfy typing
        from qdrant_client.http.models import Batch
        batch = Batch(
            ids=list(ids),
            vectors=[list(v) for v in vectors],
            payloads=list(payloads) if payloads else None,
        )
        self.client.upsert(collection_name=self.collection, points=batch)

    def search(self, query_vector: Sequence[float], top_k: int = 5, hnsw_ef: int | None = None) -> List[dict]:
        search_params = SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef is not None else None
        res = self.client.search(
            collection_name=self.collection,
            query_vector=list(query_vector),
            limit=top_k,
            search_params=search_params,
        )
        return [hit.dict() for hit in res]


