"""Read data/sources.yaml and pull each source into data/raw/<name>/.

Idempotent: skips a source whose target dir already exists and is non-empty.
Use --force to re-download.

Auth:
  roboflow   -> $ROBOFLOW_API_KEY
  kaggle     -> ~/.kaggle/kaggle.json
  huggingface-> `huggingface-cli login` (gated repos only)
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = ROOT / "data" / "sources.yaml"
RAW = ROOT / "data" / "raw"


def _is_populated(d: Path) -> bool:
    return d.exists() and any(d.iterdir())


def _git(src: dict, dest: Path) -> None:
    print(f"[git] cloning {src['url']} -> {dest}")
    subprocess.check_call(["git", "clone", "--depth", "1", src["url"], str(dest)])


def _roboflow(src: dict, dest: Path) -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set in environment")
    from roboflow import Roboflow  # type: ignore
    print(f"[roboflow] downloading {src['workspace']}/{src['project']} v{src['version']}")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(src["workspace"]).project(src["project"])
    project.version(int(src["version"])).download("yolov8", location=str(dest))


def _huggingface(src: dict, dest: Path) -> None:
    from huggingface_hub import snapshot_download  # type: ignore
    print(f"[hf] downloading {src['repo']}")
    snapshot_download(repo_id=src["repo"], repo_type="dataset", local_dir=str(dest))


def _kaggle(src: dict, dest: Path) -> None:
    print(f"[kaggle] downloading {src['handle']}")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "kaggle", "datasets", "download", "-d", src["handle"],
        "-p", str(dest), "--unzip",
    ])


def _url(src: dict, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fname = dest / "download.zip"
    print(f"[url] downloading {src['url']} -> {fname}")
    with requests.get(src["url"], stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"[url] extracting {fname}")
    with zipfile.ZipFile(fname) as z:
        z.extractall(dest)
    fname.unlink(missing_ok=True)


def _local(src: dict, dest: Path) -> None:
    src_path = ROOT / src["path"]
    if not src_path.exists():
        print(f"[local] {src_path} not present; create it and add empty .txt label "
              "files for negative examples (industrial steam/dust/welding frames). Skipping.")
        return
    if src_path.resolve() == dest.resolve():
        print(f"[local] using {dest} in place")
        return
    print(f"[local] copying {src_path} -> {dest}")
    shutil.copytree(src_path, dest, dirs_exist_ok=True)


HANDLERS = {
    "git": _git,
    "roboflow": _roboflow,
    "huggingface": _huggingface,
    "kaggle": _kaggle,
    "url": _url,
    "local": _local,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES))
    ap.add_argument("--force", action="store_true", help="re-download even if dir exists")
    ap.add_argument("--only", nargs="*", help="only fetch these source names")
    args = ap.parse_args()

    with open(args.sources) as f:
        cfg = yaml.safe_load(f)

    RAW.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for src in cfg.get("sources", []):
        name = src["name"]
        if args.only and name not in args.only:
            continue
        dest = RAW / name
        if _is_populated(dest) and not args.force:
            print(f"[skip] {name}: already present at {dest} (use --force to re-download)")
            continue
        if args.force and dest.exists():
            shutil.rmtree(dest)

        handler = HANDLERS.get(src["type"])
        if handler is None:
            print(f"[skip] {name}: unknown type '{src['type']}'")
            continue

        try:
            handler(src, dest)
            print(f"[ok] {name} -> {dest}")
        except Exception as e:
            print(f"[fail] {name}: {e}")
            failures.append(name)

    if failures:
        sys.stderr.write(f"\n[done with errors] failed sources: {failures}\n")
        return 1
    print("\n[done] all sources fetched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
