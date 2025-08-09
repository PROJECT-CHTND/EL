#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List

import redis
import requests  # type: ignore


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
APP_URL = os.getenv("APP_URL", "http://127.0.0.1:8000")


def seed_redis() -> List[Dict[str, Any]]:
    r = redis.from_url(REDIS_URL, decode_responses=True)
    hypotheses: List[Dict[str, Any]] = [
        {
            "id": "h1",
            "text": "ユーザーはクラウド移行に関心がある",
            "slots": ["cloud", "migration"],
            "belief": 0.50,
            "belief_ci": [0.4, 0.6],
            "action_cost": {"ask": 1.0, "search": 0.8},
            "status": "open",
        },
        {
            "id": "h2",
            "text": "セキュリティよりもコスト最適化を優先している",
            "slots": ["cost"],
            "belief": 0.60,
            "belief_ci": [0.5, 0.7],
            "action_cost": {"ask": 1000000.0, "search": 1000000.0},  # force stop
            "status": "open",
        },
        {
            "id": "h3",
            "text": "ベンダーロックインを避けたい",
            "slots": ["vendor", "lockin", "multi-cloud"],
            "belief": 0.70,
            "belief_ci": [0.6, 0.8],
            "action_cost": {"ask": 0.9, "search": 0.6},
            "status": "open",
        },
    ]
    r.set("e2e:hypotheses", json.dumps(hypotheses, ensure_ascii=False))
    return hypotheses


def wait_for_health(url: str, timeout_sec: int = 30) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            res = requests.get(f"{url}/health", timeout=2)
            if res.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("App health check timed out")


def start_app() -> subprocess.Popen[str]:
    # Start uvicorn server as subprocess
    env = os.environ.copy()
    # Avoid OpenAI calls during e2e
    env.setdefault("OPENAI_API_KEY", "dummy")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "el_agent.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc


def stop_app(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            proc.kill()


def main() -> int:
    original = seed_redis()

    proc = start_app()
    try:
        wait_for_health(APP_URL, timeout_sec=30)
        payload = {"user_msg": "こんにちは。要件をもう少し具体化したいです。"}
        res = requests.post(f"{APP_URL}/respond", json=payload, timeout=30)
        res.raise_for_status()
        data = res.json()
    finally:
        stop_app(proc)

    actions = data.get("actions", [])
    updated = data.get("hypotheses", [])
    markdown = data.get("markdown", "")

    # Print required outputs
    print("Selected actions:")
    for a in actions:
        print(f"- {a.get('target_hypothesis')}: action={a.get('action')}, stop={a.get('stop_rule_hit')}")

    print("\nUpdated beliefs:")
    original_map = {h["id"]: h for h in original}
    belief_increase = False
    for h in updated:
        before = float(original_map[h["id"]]["belief"]) if h["id"] in original_map else float("nan")
        after = float(h.get("belief", 0.0))
        print(f"- {h['id']}: {before:.3f} -> {after:.3f}")
        if after > before:
            belief_increase = True

    any_stop = any(bool(a.get("stop_rule_hit")) for a in actions)
    md_len = len(markdown or "")

    print(f"\nAny stop_rule_hit: {any_stop}")
    print(f"Final markdown length: {md_len}")

    if belief_increase and any_stop and md_len > 200:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())


