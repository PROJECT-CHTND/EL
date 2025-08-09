from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
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
            quantization_config=ProductQuantization(
                product=ProductQuantizationConfig(compression="x8"),
            ),
        )

    def upsert_vectors(self, ids: Sequence[int], vectors: Sequence[Sequence[float]], payloads: Optional[Sequence[dict]] = None) -> None:
        self.client.upsert(collection_name=self.collection, points={
            "ids": list(ids),
            "vectors": list(vectors),
            "payloads": list(payloads) if payloads else None,
        })

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[dict]:
        res = self.client.search(collection_name=self.collection, query_vector=list(query_vector), limit=top_k)
        return [hit.dict() for hit in res]


