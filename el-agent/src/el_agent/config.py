from functools import lru_cache
import os
from typing import Optional

from pydantic import BaseModel

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

    class Settings(BaseSettings):
        openai_api_key: Optional[str] = None

        redis_url: Optional[str] = None

        neo4j_uri: Optional[str] = None
        neo4j_user: Optional[str] = None
        neo4j_password: Optional[str] = None

        qdrant_url: Optional[str] = None

        es_url: Optional[str] = None

        # VoI / strategist parameters (M2 prep)
        voi_tau_stop: Optional[float] = None

        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @lru_cache(maxsize=1)
    def get_settings() -> Settings:
        return Settings()  # type: ignore[call-arg]

except Exception:  # pragma: no cover - fallback when pydantic_settings not installed

    class Settings(BaseModel):  # minimal fallback
        openai_api_key: Optional[str] = None
        redis_url: Optional[str] = None
        neo4j_uri: Optional[str] = None
        neo4j_user: Optional[str] = None
        neo4j_password: Optional[str] = None
        qdrant_url: Optional[str] = None
        es_url: Optional[str] = None
        voi_tau_stop: Optional[float] = None

    @lru_cache(maxsize=1)
    def get_settings() -> Settings:
        return Settings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            redis_url=os.getenv("REDIS_URL"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            qdrant_url=os.getenv("QDRANT_URL"),
            es_url=os.getenv("ES_URL"),
            voi_tau_stop=float(os.getenv("EL_VOI_TAU_STOP", "0.08")),
        )


class ServiceHealth(BaseModel):
    ok: bool = True


