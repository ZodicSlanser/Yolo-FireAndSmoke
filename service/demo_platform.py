import asyncio
import json
from collections import deque
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI(title="METI Custom Vision — demo platform")

events: deque = deque(maxlen=200)
subscribers: list[asyncio.Queue] = []


@app.post("/api/v1/internal/vision-events")
async def ingest(event: dict) -> dict:
    event["received_at"] = datetime.now(timezone.utc).isoformat()
    events.appendleft(event)
    for q in list(subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass
    return {"ok": True, "id": event.get("id")}


@app.get("/api/v1/internal/vision-events")
async def list_events(limit: int = 50) -> dict:
    return {"events": list(events)[:limit]}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "subscribers": len(subscribers), "buffered": len(events)}


@app.get("/stream")
async def stream() -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    subscribers.append(q)

    async def gen():
        try:
            for prior in list(events)[:20][::-1]:
                yield f"data: {json.dumps(prior)}\n\n"
            while True:
                ev = await q.get()
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            if q in subscribers:
                subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>METI Custom Vision — fire/smoke</title>
<style>
:root { color-scheme: dark; }
body { font-family: ui-sans-serif, system-ui, sans-serif; background:#0f172a; color:#e2e8f0;
       margin:0; padding:24px; }
header { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:16px; }
h1 { color:#f59e0b; margin:0; font-size:22px; }
.sub { color:#94a3b8; font-size:13px; }
.card { background:#1e293b; border-left:4px solid #475569; padding:12px 16px; margin:8px 0;
        border-radius:4px; box-shadow:0 1px 2px #00000033; }
.card.fire { border-color:#dc2626; }
.card.smoke { border-color:#f59e0b; }
.row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.t { color:#94a3b8; font-size:12px; }
.k { color:#7dd3fc; }
.tag { background:#0b1220; border:1px solid #334155; color:#cbd5e1;
       padding:2px 8px; border-radius:999px; font-size:11px; }
.attrib { margin-top:6px; padding:8px 10px; background:#0b1220;
          border:1px solid #1f2937; border-radius:4px; font-size:13px; }
.empty { color:#64748b; font-style:italic; padding:24px; text-align:center; }
</style></head><body>
<header>
  <h1>METI Custom Vision &mdash; fire/smoke early warning</h1>
  <span class="sub">live event feed via SSE</span>
</header>
<div id="feed"><div class="empty" id="placeholder">Waiting for events&hellip;</div></div>
<script>
const feed = document.getElementById("feed");
const placeholder = document.getElementById("placeholder");
const es = new EventSource("/stream");
es.onmessage = (e) => {
  const ev = JSON.parse(e.data);
  if (placeholder) placeholder.remove();
  const klass = (ev.type || "").endsWith("fire") ? "fire" : "smoke";
  const card = document.createElement("div");
  card.className = "card " + klass;
  const conf = (ev.confidence ?? 0).toFixed(2);
  const frames = ev.frames_in_window ?? "-";
  let attribHtml = "";
  const primary = ev.rtls_attribution && ev.rtls_attribution.primary;
  if (primary) {
    attribHtml = `<div class="attrib"><b>RTLS attribution:</b> ${primary.name} &middot;
      ${primary.role} &middot; <span class="t">${primary.id}</span></div>`;
  }
  card.innerHTML = `
    <div class="row">
      <span class="tag">${ev.type || "vision"}</span>
      <span class="t">${ev.received_at || ""}</span>
    </div>
    <div class="row" style="margin-top:6px">
      <b>cam ${ev.camera_id || "?"}</b>
      <span class="tag">zone ${ev.zone_hint || "?"}</span>
      <span>conf <span class="k">${conf}</span></span>
      <span>frames <span class="k">${frames}</span></span>
    </div>
    ${attribHtml}`;
  feed.prepend(card);
};
es.onerror = () => { /* SSE auto-reconnects */ };
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return _INDEX_HTML
