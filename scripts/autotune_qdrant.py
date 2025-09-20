#!/usr/bin/env python3
from __future__ import annotations

import os
import statistics
import time
from typing import List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams


def _p95(values: Sequence[float]) -> float:
    if not values:
        return float("inf")
    sorted_vals = sorted(values)
    k = max(0, int(len(sorted_vals) * 0.95) - 1)
    return sorted_vals[k]


def run_benchmark(qdrant_url: str, collection: str, vector_size: int, ef_values: Sequence[int], trials: int = 50, top_k: int = 16) -> int:
    client = QdrantClient(url=qdrant_url)

    # Prepare a dummy vector to avoid dependency on encoder
    query_vector = [0.0] * vector_size

    ef_to_latencies: dict[int, List[float]] = {}
    for ef in ef_values:
        latencies: List[float] = []
        for _ in range(trials):
            t0 = time.perf_counter()
            client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=ef),
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
        ef_to_latencies[ef] = latencies

    # Choose best by lowest p95 latency
    best_ef = min(ef_to_latencies.keys(), key=lambda e: _p95(ef_to_latencies[e]))
    for ef, lats in ef_to_latencies.items():
        print(f"ef={ef} p50={statistics.median(lats):.2f}ms p95={_p95(lats):.2f}ms n={len(lats)}")

    return best_ef


def main() -> int:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection = os.getenv("QDRANT_COLLECTION", "el_vec")
    vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "1536"))
    ef_values = [64, 100, 150, 200, 256]

    best_ef = run_benchmark(qdrant_url, collection, vector_size, ef_values)

    repo_root = os.path.dirname(__file__)
    cfg_dir = os.path.join(os.path.dirname(repo_root), "config")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "qdrant_ef.txt")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(str(best_ef))
    print(f"[Qdrant] Best hnsw_ef={best_ef} written to {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


