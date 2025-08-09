from __future__ import annotations

from typing import Optional

import redis


class RedisStore:
    def __init__(self, url: str):
        self._url = url
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self._url, decode_responses=True)
        return self._client

    def set_json(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self.client.set(name=key, value=value, ex=ex)

    def get_json(self, key: str) -> Optional[str]:
        return self.client.get(name=key)


