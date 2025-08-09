#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Any, Dict


def create_elasticsearch_index(es_url: str, index_name: str) -> None:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import TransportError

    client = Elasticsearch(es_url)

    try:
        if client.indices.exists(index=index_name):  # type: ignore[attr-defined]
            print(f"[ES] Index '{index_name}' already exists – skipping")
            return
    except Exception as e:  # pragma: no cover
        print(f"[ES] Failed to check index existence: {e}")
        raise

    settings: Dict[str, Any] = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "default": {"type": "standard"}
                }
            },
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "content": {"type": "text"},
                "lang": {"type": "keyword"},
                "url": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
            }
        },
    }

    try:
        client.indices.create(index=index_name, body=settings)  # type: ignore[attr-defined]
        print(f"[ES] Created index '{index_name}'")
    except TransportError as te:  # pragma: no cover
        if te.error == "resource_already_exists_exception":
            print(f"[ES] Index '{index_name}' already exists – skipping")
        else:
            print(f"[ES] Failed to create index: {te}")
            raise


def create_qdrant_collection(qdrant_url: str, collection_name: str, vector_size: int) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        VectorParams,
        HnswConfigDiff,
        OptimizersConfigDiff,
        ProductQuantization,
        ProductQuantizationConfig,
    )

    client = QdrantClient(url=qdrant_url)

    collections = {c.name for c in client.get_collections().collections}
    if collection_name in collections:
        print(f"[Qdrant] Collection '{collection_name}' already exists – skipping")
        return

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=10000),
        quantization_config=ProductQuantization(
            product=ProductQuantizationConfig(compression="x8")
        ),
    )
    print(f"[Qdrant] Created collection '{collection_name}' (dim={vector_size})")


def main() -> int:
    es_url = os.getenv("ES_URL", "http://localhost:9200")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_dim_str = os.getenv("QDRANT_VECTOR_SIZE", "1536")

    try:
        qdrant_dim = int(qdrant_dim_str)
    except ValueError:
        print(f"[Qdrant] Invalid QDRANT_VECTOR_SIZE='{qdrant_dim_str}', expected int")
        return 2

    # Elasticsearch index
    create_elasticsearch_index(es_url=es_url, index_name="el_docs")

    # Qdrant collection
    create_qdrant_collection(qdrant_url=qdrant_url, collection_name="el_vec", vector_size=qdrant_dim)

    # Verification – fail-fast if not present
    try:
        from elasticsearch import Elasticsearch
        es_client = Elasticsearch(es_url)
        assert es_client.indices.exists(index="el_docs")  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        print(f"[ES] Verification failed: {e}")
        return 3

    try:
        from qdrant_client import QdrantClient
        q_client = QdrantClient(url=qdrant_url)
        names = {c.name for c in q_client.get_collections().collections}
        assert "el_vec" in names
    except Exception as e:  # pragma: no cover
        print(f"[Qdrant] Verification failed: {e}")
        return 4

    print("Done: indices are ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())


