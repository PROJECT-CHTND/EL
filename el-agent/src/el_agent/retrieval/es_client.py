from __future__ import annotations

from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch


class ESClient:
    def __init__(self, url: str):
        self.client = Elasticsearch(url)

    def search_bm25(self, index: str, query: str, size: int = 10) -> List[Dict[str, Any]]:
        body = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "operator": "and",
                    }
                }
            }
        }
        res = self.client.search(index=index, body=body, size=size)
        hits = res.get("hits", {}).get("hits", [])
        return hits


