"""Pull a few short industrial fire/smoke videos via yt-dlp into demo-clips/.

URLs are placeholders — replace with whatever clips you want for the demo.
Edit URLS below; everything else is just yt-dlp invocation.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo-clips"

URLS: list[str] = [
    # Add 3–5 YouTube URLs of industrial fire/smoke footage for the demo.
    # Examples (verify before using):
    # "https://www.youtube.com/watch?v=YOUR_CLIP_1",
    # "https://www.youtube.com/watch?v=YOUR_CLIP_2",
]


def main() -> int:
    if shutil.which("yt-dlp") is None:
        sys.stderr.write("[fail] yt-dlp not on PATH. Install with `pip install yt-dlp`.\n")
        return 1
    if not URLS:
        sys.stderr.write(
            "[info] no URLs configured. Edit scripts/download_sample_clips.py and add\n"
            "       3–5 YouTube URLs to the URLS list, then re-run.\n"
        )
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    pattern = str(OUT / "clip_%(autonumber)s.%(ext)s")
    cmd = ["yt-dlp", "-f", "best[ext=mp4][height<=720]", "-o", pattern, *URLS]
    print(f"[info] running: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
