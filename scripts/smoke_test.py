"""Run yolo11n.pt on a sample image. Proves Ultralytics + OpenCV + weights download work."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"


def main() -> int:
    try:
        import torch
        from ultralytics import YOLO
    except Exception as e:
        sys.stderr.write(f"[fail] cannot import torch/ultralytics: {e}\n")
        return 1

    print(f"[info] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] cuda device: {torch.cuda.get_device_name(0)}")

    out_dir = ROOT / "runs" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] loading yolo11n.pt (downloads on first run)…")
    try:
        model = YOLO("yolo11n.pt")
    except Exception as e:
        sys.stderr.write(f"[fail] could not load yolo11n.pt: {e}\n")
        sys.stderr.write(
            "If this is a version-pin problem, try: pip install --upgrade ultralytics\n"
        )
        return 1

    print(f"[info] running inference on {SAMPLE_URL}")
    try:
        results = model.predict(SAMPLE_URL, save=True, project=str(out_dir), name="bus", exist_ok=True)
    except Exception as e:
        sys.stderr.write(f"[fail] inference error: {e}\n")
        return 1

    r = results[0]
    counts: dict[str, int] = {}
    for box in r.boxes:
        cls = r.names[int(box.cls)]
        counts[cls] = counts.get(cls, 0) + 1

    print(f"[ok] detected: {counts}")
    print(f"[ok] annotated image saved under {out_dir}/bus/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
