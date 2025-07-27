from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import jwt  # type: ignore
import requests  # type: ignore
from fastapi import Depends, HTTPException, Security, status  # type: ignore
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # type: ignore

ALGORITHM = "RS256"

auth_scheme = HTTPBearer()


class AuthSettings:
    def __init__(self) -> None:
        self.domain = os.getenv("AUTH0_DOMAIN")
        self.audience = os.getenv("AUTH0_AUDIENCE")
        if not self.domain or not self.audience:
            raise RuntimeError("AUTH0_DOMAIN and AUTH0_AUDIENCE must be set for JWT auth")

    @property
    def jwks_url(self) -> str:  # noqa: D401
        return f"https://{self.domain}/.well-known/jwks.json"


@lru_cache
def get_jwks() -> Dict[str, Any]:
    settings = AuthSettings()
    resp = requests.get(settings.jwks_url, timeout=5)
    resp.raise_for_status()
    return resp.json()


def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)) -> Dict[str, Any]:  # noqa: D401
    token = credentials.credentials
    unverified_header = jwt.get_unverified_header(token)
    jwks = get_jwks()
    key = next((k for k in jwks["keys"] if k["kid"] == unverified_header["kid"]), None)
    if key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header")

    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
    try:
        decoded = jwt.decode(
            token,
            public_key,
            algorithms=[ALGORITHM],
            audience=AuthSettings().audience,
            issuer=f"https://{AuthSettings().domain}/",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    return decoded 