from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


_TRACE_ENV = os.getenv("EL_TRACE", "0")
_TRACE_DIR = Path(os.getenv("EL_TRACE_DIR", str(Path(__file__).resolve().parents[2] / "logs" / "wal")))

# Thread-local buffer for per-request/session metadata if needed later
_local = threading.local()


def trace_enabled() -> bool:
    return _TRACE_ENV == "1"


def _ensure_dir() -> None:
    try:
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _to_serializable(obj: Any) -> Any:
    # Best-effort conversion for Pydantic v2 models
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump(exclude_none=True)
        if hasattr(obj, "model_dump_json"):
            # keep as parsed dict instead of JSON string
            try:
                return json.loads(obj.model_dump_json())
            except Exception:
                return obj.model_dump_json()
    except Exception:
        pass

    # Collections
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [_to_serializable(v) for v in obj]
        return t if isinstance(obj, list) else tuple(t)

    # Fallback primitives
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _mask_if_available(data: Any) -> Any:
    # Try to import PII masker from el-agent; fallback to identity
    try:
        from el_agent.utils.pii import mask_structure  # type: ignore
    except Exception:
        def mask_structure(x: Any) -> Any:  # type: ignore
            return x
    try:
        return mask_structure(data)
    except Exception:
        return data


def trace_event(stage: str, kind: str, payload: Dict[str, Any] | Any, *, meta: Dict[str, Any] | None = None) -> None:
    if not trace_enabled():
        return
    _ensure_dir()

    entry: Dict[str, Any] = {
        "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "epoch_ms": int(time.time() * 1000),
        "stage": stage,
        "kind": kind,
    }

    try:
        serial = _to_serializable(payload)
        entry["payload"] = _mask_if_available(serial)
    except Exception:
        # never fail the main flow
        entry["payload"] = "<unserializable>"

    if meta:
        try:
            entry["meta"] = _mask_if_available(_to_serializable(meta))
        except Exception:
            entry["meta"] = meta

    try:
        fname = _TRACE_DIR / ("trace_" + datetime.utcnow().strftime("%Y%m%d") + ".log")
        with fname.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Swallow any IO error
        pass


