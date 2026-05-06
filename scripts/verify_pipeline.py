"""Model-independent end-to-end verification of the demo platform.

Spawns uvicorn in a subprocess, POSTs synthetic detection events, confirms they appear
on the SSE stream, then tears down. Exits 0 on success.
"""
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

import requests

BASE = "http://localhost:8001"
INGEST = f"{BASE}/api/v1/internal/vision-events"
STREAM = f"{BASE}/stream"
HEALTHZ = f"{BASE}/healthz"
HOME = f"{BASE}/"

SAMPLE_EVENTS = [
    {
        "type": "vision.fire",
        "camera_id": "DEMO_CAM_42",
        "zone_hint": "production_line_3",
        "confidence": 0.91,
        "frames_in_window": 7,
        "rtls_attribution": {
            "method": "rtls_zone_match",
            "primary": {"id": "EMP-1142", "name": "Khalid M.", "role": "Line Supervisor"},
            "candidates": [
                {"id": "EMP-1142", "name": "Khalid M.", "role": "Line Supervisor"},
                {"id": "EMP-2034", "name": "Ahmed S.", "role": "Operator"},
            ],
        },
    },
    {
        "type": "vision.smoke",
        "camera_id": "DEMO_CAM_07",
        "zone_hint": "warehouse_a",
        "confidence": 0.74,
        "frames_in_window": 6,
        "rtls_attribution": {
            "method": "rtls_zone_match",
            "primary": {"id": "EMP-3201", "name": "Yusuf K.", "role": "Forklift Operator"},
            "candidates": [],
        },
    },
    {
        "type": "vision.smoke",
        "camera_id": "DEMO_CAM_42",
        "zone_hint": "production_line_3",
        "confidence": 0.66,
        "frames_in_window": 5,
        "rtls_attribution": {"method": "rtls_zone_match", "primary": None, "candidates": []},
    },
]


def _wait_ready(timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(HEALTHZ, timeout=1.0)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25)
    return False


def _post_events() -> list[str]:
    ids: list[str] = []
    for ev in SAMPLE_EVENTS:
        ev_id = str(uuid.uuid4())
        body = dict(ev, id=ev_id, source="meti.custom_vision.firesmoke",
                    model_version="verify_pipeline.py",
                    timestamp=datetime.now(timezone.utc).isoformat())
        r = requests.post(INGEST, json=body, timeout=2.0)
        r.raise_for_status()
        ids.append(ev_id)
        print(f"[ok] POST {body['type']} -> 200 (id={ev_id[:8]})")
    return ids


def _read_stream(expected_ids: list[str], timeout_s: float = 5.0) -> set[str]:
    seen: set[str] = set()
    expected = set(expected_ids)
    deadline = time.time() + timeout_s
    with requests.get(STREAM, stream=True, timeout=timeout_s + 1) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if time.time() > deadline:
                break
            if not raw or not raw.startswith("data: "):
                continue
            try:
                ev = json.loads(raw[len("data: "):])
            except json.JSONDecodeError:
                continue
            if ev.get("id") in expected:
                seen.add(ev["id"])
                if seen >= expected:
                    break
    return seen


def main() -> int:
    print("[info] starting uvicorn service.demo_platform:app on :8001")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "service.demo_platform:app",
         "--host", "127.0.0.1", "--port", "8001", "--log-level", "warning"],
    )
    try:
        if not _wait_ready():
            sys.stderr.write("[fail] platform did not become ready in time\n")
            return 1
        print(f"[ok] platform ready at {HOME}")

        ids = _post_events()
        seen = _read_stream(ids, timeout_s=5.0)

        missing = set(ids) - seen
        if missing:
            sys.stderr.write(f"[fail] {len(missing)} events did not arrive on /stream: {missing}\n")
            return 1

        print(f"[ok] all {len(ids)} events round-tripped through ingest -> SSE")
        print("[ok] open http://localhost:8001/ in a browser to see the dashboard "
              "(re-run verify_pipeline.py while it's open to watch live cards appear)")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
