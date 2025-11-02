from __future__ import annotations

import json
from typing import Any


def sanitize_json_content(raw: str) -> str:
    """Best-effort to strip markdown fences and leading noise, keeping JSON.

    - Removes leading ``` or ```json fences and trailing backticks.
    - Trims whitespace.
    - If residual prefix exists before the first '{' or '[', slice from there.
    """
    s = (raw or "").strip()
    if s.startswith("```json"):
        s = s[7:].strip()
        s = s.rstrip("`").strip()
    elif s.startswith("```"):
        s = s[3:].strip()
        s = s.rstrip("`").strip()

    # Heuristic: start from first JSON-looking character
    first_obj = s.find("{")
    first_arr = s.find("[")
    idxs = [i for i in (first_obj, first_arr) if i != -1]
    if idxs:
        start = min(idxs)
        s = s[start:]
    return s


def parse_json_strict(raw: str) -> Any:
    """Sanitize then parse JSON. Raises json.JSONDecodeError on failure."""
    s = sanitize_json_content(raw)
    return json.loads(s)


