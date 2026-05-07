import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Allow running both as `python -m service.inference_service` (from project root)
# and `python service/inference_service.py` (direct invocation). The direct case
# only puts service/ on sys.path; we need the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import requests
from ultralytics import YOLO

from service.config import config
from service.rtls_mock import lookup as rtls_lookup


def _emit(event: dict) -> None:
    try:
        r = requests.post(config.webhook_url, json=event, timeout=config.webhook_timeout_s)
        print(f"[webhook] -> {r.status_code} {event['type']} conf={event['confidence']:.2f}")
    except requests.RequestException as e:
        print(f"[webhook ERR] {e} (event id {event['id']})")


def _build_event(cls: str, conf: float, frames: int) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "type": f"vision.{cls}",
        "source": "meti.custom_vision.firesmoke",
        "model_version": config.model_version,
        "camera_id": config.camera_id,
        "zone_hint": config.zone_hint,
        "confidence": conf,
        "frames_in_window": frames,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rtls_attribution": rtls_lookup(config.zone_hint),
    }


def main() -> int:
    if not config.model_path.exists():
        sys.stderr.write(
            "\n[fatal] No trained weights at "
            f"{config.model_path}.\n"
            "Place a trained `best.pt` from the training step (Phase B / "
            "notebooks/03-train-yolo11.ipynb) under models/best.pt, or use\n"
            "`python scripts/verify_pipeline.py` to verify the platform "
            "wiring without a model.\n\n"
        )
        return 1

    model = YOLO(str(config.model_path))
    class_names = {int(k): v for k, v in model.names.items()}

    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        sys.stderr.write(f"[fatal] cannot open source: {config.source}\n")
        return 1

    recent: deque = deque()
    last_fired: dict[str, float] = {}

    print(
        f"[info] model={config.model_path.name} classes={list(class_names.values())} "
        f"source={config.source} conf>={config.conf_threshold}"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        results = model.predict(frame, conf=config.conf_threshold, imgsz=config.imgsz, verbose=False)[0]

        per_class_max_conf: dict[str, float] = {}
        for box in results.boxes:
            cls_idx = int(box.cls)
            cls = class_names.get(cls_idx, str(cls_idx))
            conf = float(box.conf)
            recent.append((now, cls, conf))
            per_class_max_conf[cls] = max(per_class_max_conf.get(cls, 0.0), conf)

        while recent and now - recent[0][0] > config.temporal_window_s:
            recent.popleft()

        counts: dict[str, int] = {}
        for _, c, _ in recent:
            counts[c] = counts.get(c, 0) + 1

        for cls, n in counts.items():
            if n < config.temporal_frames:
                continue
            if now - last_fired.get(cls, 0.0) < config.refire_cooldown_s:
                continue
            best_conf = max(
                (c for ts, cl, c in recent if cl == cls),
                default=per_class_max_conf.get(cls, 0.0),
            )
            event = _build_event(cls, best_conf, n)
            _emit(event)
            last_fired[cls] = now

        if config.show_window:
            annotated = results.plot()
            cv2.imshow("METI Custom Vision — fire/smoke", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cap.release()
    if config.show_window:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
