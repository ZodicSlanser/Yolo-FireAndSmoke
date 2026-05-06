from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DemoConfig:
    model_path: Path = ROOT / "models" / "best.pt"
    model_version: str = "v11s-2026.05.06"

    conf_threshold: float = 0.45
    imgsz: int = 640
    source: str = str(ROOT / "demo-clips" / "clip_001.mp4")
    camera_id: str = "DEMO_CAM_42"
    zone_hint: str = "production_line_3"

    temporal_frames: int = 5
    temporal_window_s: float = 4.0
    refire_cooldown_s: float = 30.0

    webhook_url: str = "http://localhost:8001/api/v1/internal/vision-events"
    webhook_timeout_s: float = 2.0

    show_window: bool = True


config = DemoConfig()
