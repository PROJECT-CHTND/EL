from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, Iterable


def _get_salt() -> bytes:
    return os.getenv("PII_SALT", "").encode("utf-8")


def _hash_token(kind: str, value: str, length: int = 10) -> str:
    h = hashlib.sha256()
    h.update(_get_salt())
    h.update(f"{kind}|".encode("utf-8"))
    h.update(value.encode("utf-8"))
    return h.hexdigest()[:length]


# --- Regex Patterns ---
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Match phone-like sequences, filtered by digit count >= 10 later
PHONE_RE = re.compile(r"\+?\d[\d\-\.\s()]{8,}\d")

# Simple street address heuristic (English-like). This is intentionally conservative.
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,6}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pkwy|Parkway|Pl|Place|Sq|Square)\b\.?",
    re.IGNORECASE,
)


def mask_text(text: str) -> str:
    if not text:
        return text

    # Emails
    def _email_sub(m: re.Match[str]) -> str:
        token = _hash_token("email", m.group(0))
        return f"<email:{token}>"

    masked = EMAIL_RE.sub(_email_sub, text)

    # Phones (require >=10 digits to reduce false positives)
    def _phone_sub(m: re.Match[str]) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D+", "", raw)
        if len(digits) < 10:
            return raw
        token = _hash_token("phone", digits)
        return f"<phone:{token}>"

    masked = PHONE_RE.sub(_phone_sub, masked)

    # Addresses
    def _addr_sub(m: re.Match[str]) -> str:
        raw = m.group(0)
        # Normalize spaces for hashing
        norm = re.sub(r"\s+", " ", raw.strip())
        token = _hash_token("address", norm.lower())
        return f"<address:{token}>"

    masked = ADDRESS_RE.sub(_addr_sub, masked)
    return masked


def mask_structure(data: Any) -> Any:
    """Recursively mask any strings within dict/list structures.

    - Strings are passed through mask_text
    - Dicts/lists/tuples are traversed
    - Other scalars are returned as-is
    """
    if data is None:
        return None
    if isinstance(data, str):
        return mask_text(data)
    if isinstance(data, dict):
        return {k: mask_structure(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        t = [mask_structure(v) for v in data]
        return type(data)(t) if isinstance(data, tuple) else t
    return data


def user_id_hash(session_id: str | None) -> str | None:
    if not session_id:
        return None
    h = hashlib.sha256()
    h.update(_get_salt())
    h.update(b"|session|")
    h.update(session_id.encode("utf-8"))
    return h.hexdigest()[:16]



