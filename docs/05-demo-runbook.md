# 05 — Demo runbook

Day-of script for the BANDARIYAH / Protex narrative. Runs on the **4090 laptop**.

---

## 5 minutes before

1. Two terminals open in `D:/Work/METI/PoCs/Yolo-FireAndSmoke/` with `.venv` activated.
2. Browser open to `http://localhost:8001/`. Full-screen.
3. **Terminal 1**: `uvicorn service.demo_platform:app --port 8001` — leave running.
4. **Terminal 2**: have `python service/inference_service.py` queued, don't run yet.
5. Confirm `models/best.pt` exists. Confirm `demo-clips/clip_001.mp4` exists (or whichever clip you've set as `source` in `config.py`).
6. Test once: hit the URL, refresh, make sure the page loads. Run `python scripts/verify_pipeline.py` to confirm cards appear; close it down.

---

## The narrative (≈3 minutes)

**Setup (30 s)**

> "Protex AI handles PPE, intrusion, behaviour, vehicle-pedestrian. What it doesn't handle is smoke and fire — that's a real gap, and the customer's tender requires it. Rather than swap vendors, we built it."

**Demo (90 s)**

> "This is YOLOv11 running on an edge GPU class device, watching a high-risk subset of cameras. Here's a real industrial fire video being fed in as if it were a live RTSP stream..."

Hit play in Terminal 2. OpenCV window shows annotated frames. Watch the confidence numbers climb past threshold; a few seconds later the temporal-confirmation rule trips and a card appears in the dashboard.

**Attribution (45 s)**

> "Notice the card. We don't just say 'smoke detected.' We say smoke detected on `production_line_3`, and we attribute the event to a person via RTLS — Khalid M., Line Supervisor, present in zone at the time. This is what the existing fire panel can't do. The panel knows there's smoke; it doesn't know who was there."

**Compliance close (15 s)**

> "We're explicit that this is supplementary detection, not code-compliant primary fire detection. Your existing fire & safety system stays primary. We add zone-attributed early warning and HSE evidence on top."

---

## If something goes wrong

- **No webhook arrives** → Terminal 1 isn't running, or the URL doesn't match. Check `webhook_url` in `service/config.py`.
- **Detection looks bad** → swap to a different clip. Always have 3+ ready in `demo-clips/`. Don't try to retrain or tune live.
- **GPU OOM** → drop `imgsz` in `config.py` to 480, restart. Shouldn't happen on the 4090.
- **OpenCV window is black** → some Windows OpenCV builds fight with hardware-decode video. Re-encode the clip with `ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`.
- **Browser dashboard doesn't update** → SSE connection dropped. Refresh the page.
- **Whole laptop feels hot / fan loud** → power profile is on; that's normal during inference. Don't switch to battery saver.

---

## What this demo proves and what it doesn't

**Proves:**

- Fire/smoke detection at credible accuracy from CCTV-style video, in real time.
- Our platform consumes the events through the same path it consumes Protex events.
- Zone attribution + RTLS-based identity attribution, which the fire panel cannot do.
- Cheap bill of materials — one small GPU appliance, no recurring vendor cost.

**Does not prove:**

- Production performance in BANDARIYAH-specific lighting / processes (Phase 1/2 site-tuning).
- Code-compliant primary fire detection (proposal §9.4 explicitly disclaims this).
- Multi-camera fusion at scale (architecture supports it via the same Rule Engine).

When asked "is this certified for fire detection?" — **"No, and we don't claim to be. The customer's existing fire & safety system is your primary, code-compliant detection. We add zone-attributed early warning and HSE evidence on top of it."**
