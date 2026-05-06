# 04 — Drop-in trained model

When training is done (Phase B), this is what changes.

## Steps

1. Copy `runs/firesmoke/v11s-neg-tuned/weights/best.pt` (or whichever run you want to deploy) into `models/best.pt`. Notebook 03 does this automatically in its last cell.

2. (Optional) Pull industrial fire footage for the demo:
   ```bash
   # Edit scripts/download_sample_clips.py with 3-5 YouTube URLs first
   python scripts/download_sample_clips.py
   ```
   Or drop your own .mp4 files into `demo-clips/`.

3. Point `service/config.py` at a clip:
   ```python
   source: str = str(ROOT / "demo-clips" / "clip_001.mp4")
   ```

4. Run the demo:
   ```bash
   # Terminal 1
   uvicorn service.demo_platform:app --port 8001
   # Terminal 2
   python service/inference_service.py
   ```

   OpenCV window shows annotated frames. Browser dashboard at http://localhost:8001/ shows confirmed events with RTLS attribution.

This is the **first time the full inference loop runs**. Until trained weights existed, there was nothing meaningful to detect — the synthetic-event verification (`scripts/verify_pipeline.py`) covered the platform side.

---

## Tuning

If detections look bad on real industrial footage:

| Symptom | Knob | Direction |
|---|---|---|
| Too many false positives (welding sparks, sunlight) | `conf_threshold` | 0.45 → 0.55 → 0.65 |
| Confirmed events fire on transient flicker | `temporal_frames` | 5 → 8 |
| Missed early-stage smoke | `conf_threshold` | 0.45 → 0.35 (don't go below 0.30) |
| Missed early-stage smoke + don't want FPs | Train better. Add smoke-only sources, fine-tune. |
| Same event keeps re-firing | `refire_cooldown_s` | 30 → 60 |

If still bad after threshold tuning, the model needs more training data of the kind you're missing on — not a knob change. Run the **negative-class fine-tune** in [`02-training-guide.md`](02-training-guide.md#stage-3--industrial-false-positive-suppression) with site-specific footage.

---

## Real RTSP source

The inference service uses `cv2.VideoCapture(config.source)` which accepts any URL OpenCV understands. Point `config.source` at:

```python
source: str = "rtsp://user:pass@cam.local/stream"
```

For production, wrap the read loop in a reconnect-on-failure block (5-second backoff) — RTSP streams drop and rejoin all the time.

---

## Multi-camera

Run multiple `inference_service.py` processes with different `camera_id` / `zone_hint` / `source`, all POSTing to the same `demo_platform` ingest. The dashboard already handles arbitrary-cardinality streams. Easiest way: set those values via env vars and read them in `service/config.py`.
