"""Merge raw datasets in data/raw/ into a unified train/val/test split.

Steps: normalize (rewrite class indices to unified [fire=0, smoke=1]) -> dedupe
(perceptual hash, within-source then cross-source) -> group-aware stratified
split -> apply per-source train weights -> write data/merged/{train,val,test}/
and data.yaml -> emit PREP_REPORT.md.

Run with --no-dedup --no-weights for a fast first pass.
"""
import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image
import imagehash
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES = ROOT / "data" / "sources.yaml"
RAW = ROOT / "data" / "raw"
STAGING = ROOT / "data" / "staging"
DEFAULT_OUT = ROOT / "data" / "merged"

UNIFIED_NAMES = ["fire", "smoke"]
NAME_TO_IDX = {n: i for i, n in enumerate(UNIFIED_NAMES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FRAME_RE = re.compile(r"(.+?)[_-]frame[_-]?\d+", re.IGNORECASE)


# ---------- discovery ----------

def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Walk a YOLO-format source dir; return (image, label) pairs.

    Tolerant of layouts:
      <root>/{train,valid,val,test}/images/*.jpg + .../labels/*.txt
      <root>/images/*.jpg + <root>/labels/*.txt
      <root>/*.jpg + <root>/*.txt
    """
    pairs: list[tuple[Path, Path]] = []
    for img in root.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS or not img.is_file():
            continue
        candidates = [
            img.with_suffix(".txt"),
            img.parent.parent / "labels" / (img.stem + ".txt"),
        ]
        for cand in candidates:
            if cand.exists():
                pairs.append((img, cand))
                break
    return pairs


# ---------- stage 1: normalize ----------

def _stage_source(src: dict) -> tuple[Path, dict]:
    name = src["name"]
    src_root = RAW / name
    out_dir = STAGING / name
    out_imgs = out_dir / "images"
    out_lbls = out_dir / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    class_map = {int(k): v for k, v in (src.get("class_map") or {}).items()}
    pairs = _find_pairs(src_root)
    counts = Counter()
    written = 0

    for img, lbl in tqdm(pairs, desc=f"normalize {name}", leave=False):
        new_img = out_imgs / f"{name}_{img.stem}{img.suffix}"
        new_lbl = out_lbls / f"{name}_{img.stem}.txt"
        if not new_img.exists():
            shutil.copy2(img, new_img)

        out_lines: list[str] = []
        with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    src_idx = int(parts[0])
                except ValueError:
                    continue
                unified_name = class_map.get(src_idx)
                if unified_name is None:
                    continue
                if unified_name not in NAME_TO_IDX:
                    continue
                new_idx = NAME_TO_IDX[unified_name]
                out_lines.append(" ".join([str(new_idx)] + parts[1:]))
                counts[unified_name] += 1
        with open(new_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))
            if out_lines:
                f.write("\n")
        written += 1
        if not out_lines:
            counts["negative"] += 1

    return out_dir, {"name": name, "images": written, "labels": dict(counts)}


# ---------- stage 2: dedupe ----------

def _phash(path: Path) -> int | None:
    try:
        with Image.open(path) as im:
            return int(str(imagehash.phash(im)), 16)
    except Exception:
        return None


def _dedupe(staged: list[Path], threshold: int = 4) -> set[Path]:
    """Return the set of image paths to keep after perceptual-hash dedup."""
    all_imgs: list[Path] = []
    for s in staged:
        all_imgs += sorted((s / "images").glob("*"))

    print(f"[dedupe] hashing {len(all_imgs)} images")
    hashed: list[tuple[Path, int]] = []
    for p in tqdm(all_imgs, desc="phash", leave=False):
        h = _phash(p)
        if h is not None:
            hashed.append((p, h))

    keep: list[tuple[Path, int]] = []
    dropped = 0
    for p, h in hashed:
        is_dup = False
        for _, kh in keep:
            if bin(h ^ kh).count("1") <= threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
        else:
            keep.append((p, h))
    print(f"[dedupe] kept {len(keep)} / dropped {dropped}")
    return {p for p, _ in keep}


# ---------- stage 3: split ----------

def _group_key(img: Path) -> str:
    stem = img.stem
    m = FRAME_RE.match(stem)
    if m:
        return f"{img.parent.parent.name}::{m.group(1)}"
    return f"{img.parent.parent.name}::{stem}"


def _strata_key(label_path: Path) -> str:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return "negative"
    has = set()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            if 0 <= idx < len(UNIFIED_NAMES):
                has.add(UNIFIED_NAMES[idx])
    if has == {"fire"}:
        return "fire_only"
    if has == {"smoke"}:
        return "smoke_only"
    if has == {"fire", "smoke"}:
        return "both"
    return "negative"


def _three_way_split(items: list[Path], groups: list[str], strata: list[str],
                     train: float, val: float, test: float, seed: int):
    """Group-aware stratified split into (train, val, test) lists of indices."""
    n = len(items)
    if test > 0:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
        idx = list(range(n))
        train_val_idx, test_idx = next(gss1.split(idx, groups=groups))
    else:
        train_val_idx = list(range(n))
        test_idx = []

    if val > 0 and len(train_val_idx) > 0:
        rel_val = val / (train + val)
        groups_tv = [groups[i] for i in train_val_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
        sub_train, sub_val = next(gss2.split(train_val_idx, groups=groups_tv))
        train_idx = [train_val_idx[i] for i in sub_train]
        val_idx = [train_val_idx[i] for i in sub_val]
    else:
        train_idx = list(train_val_idx)
        val_idx = []

    return train_idx, val_idx, list(test_idx)


# ---------- stage 4: write ----------

def _emit_split(name: str, indices: list[int], imgs: list[Path], lbls: list[Path],
                out: Path) -> dict:
    img_dir = out / name / "images"
    lbl_dir = out / name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for i in indices:
        shutil.copy2(imgs[i], img_dir / imgs[i].name)
        if lbls[i].exists():
            shutil.copy2(lbls[i], lbl_dir / lbls[i].name)
            counts[_strata_key(lbls[i])] += 1
        else:
            counts["negative"] += 1
    return dict(counts)


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--phash-threshold", type=int, default=4)
    args = ap.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-3:
        raise SystemExit("--train + --val + --test must sum to 1.0")

    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with open(args.sources) as f:
        cfg = yaml.safe_load(f)

    sources = cfg.get("sources", [])
    if not sources:
        raise SystemExit("no sources configured in sources.yaml")

    # Stage 1: normalize each source under data/staging/<name>/
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)
    norm_reports: list[dict] = []
    staged_dirs: list[Path] = []
    for src in sources:
        src_root = RAW / src["name"]
        if not src_root.exists():
            print(f"[skip] {src['name']}: data/raw/{src['name']}/ not present "
                  "(run download_datasets.py first)")
            continue
        d, rep = _stage_source(src)
        staged_dirs.append(d)
        norm_reports.append(rep)

    if not staged_dirs:
        raise SystemExit("no sources staged; nothing to merge")

    # Collect all (image, label) pairs across staged sources
    all_imgs: list[Path] = []
    all_lbls: list[Path] = []
    for s in staged_dirs:
        for img in sorted((s / "images").iterdir()):
            lbl = (s / "labels") / (img.stem + ".txt")
            all_imgs.append(img)
            all_lbls.append(lbl)
    print(f"[merge] {len(all_imgs)} pairs across {len(staged_dirs)} sources")

    # Stage 2: dedupe
    drop_summary = {}
    if not args.no_dedup:
        keep_set = _dedupe(staged_dirs, threshold=args.phash_threshold)
        before = len(all_imgs)
        kept_idx = [i for i, p in enumerate(all_imgs) if p in keep_set]
        all_imgs = [all_imgs[i] for i in kept_idx]
        all_lbls = [all_lbls[i] for i in kept_idx]
        drop_summary = {"before": before, "after": len(all_imgs)}
        print(f"[dedupe] {before} -> {len(all_imgs)}")

    # Stage 3: split
    groups = [_group_key(p) for p in all_imgs]
    strata = [_strata_key(p) for p in all_lbls]
    train_idx, val_idx, test_idx = _three_way_split(
        all_imgs, groups, strata, args.train, args.val, args.test, args.seed)
    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Stage 4: source-weight downsample (train only)
    if not args.no_weights:
        rng = random.Random(args.seed)
        weights = {s["name"]: float(s.get("weight", 1.0)) for s in sources}
        kept: list[int] = []
        for i in train_idx:
            src_name = all_imgs[i].name.split("_")[0]
            for s in sources:
                if all_imgs[i].name.startswith(s["name"] + "_"):
                    src_name = s["name"]
                    break
            w = weights.get(src_name, 1.0)
            if rng.random() < w:
                kept.append(i)
        before = len(train_idx)
        train_idx = kept
        print(f"[weights] train downsampled {before} -> {len(train_idx)}")

    # Stage 5: write splits
    splits_summary = {}
    splits_summary["train"] = _emit_split("train", train_idx, all_imgs, all_lbls, out)
    splits_summary["val"] = _emit_split("val", val_idx, all_imgs, all_lbls, out)
    if test_idx:
        splits_summary["test"] = _emit_split("test", test_idx, all_imgs, all_lbls, out)

    # Leakage check
    seen: dict[str, str] = {}
    leaks: list[tuple[str, str, str]] = []
    for split_name, idx_list in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        for i in idx_list:
            g = groups[i]
            if g in seen and seen[g] != split_name:
                leaks.append((g, seen[g], split_name))
            seen[g] = split_name

    # data.yaml
    data_yaml = {
        "path": str(out),
        "train": "train/images",
        "val": "val/images",
        "nc": len(UNIFIED_NAMES),
        "names": UNIFIED_NAMES,
        "sources_used": [r["name"] for r in norm_reports],
        "seed": args.seed,
        "splits": {"train": args.train, "val": args.val, "test": args.test},
        "deduped": not args.no_dedup,
        "weights_applied": not args.no_weights,
    }
    if test_idx:
        data_yaml["test"] = "test/images"
    with open(out / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    # PREP_REPORT.md
    report_lines = [
        "# Dataset prep report",
        "",
        f"Output: `{out}`",
        f"Seed: `{args.seed}`",
        f"Splits: train={args.train} val={args.val} test={args.test}",
        f"Deduped: {not args.no_dedup} (threshold={args.phash_threshold})",
        f"Source weights applied: {not args.no_weights}",
        "",
        "## Per-source normalization",
        "",
        "| source | images | fire labels | smoke labels | negative |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in norm_reports:
        lbls = r["labels"]
        report_lines.append(
            f"| {r['name']} | {r['images']} | {lbls.get('fire', 0)} | "
            f"{lbls.get('smoke', 0)} | {lbls.get('negative', 0)} |"
        )

    report_lines += ["", "## Splits", "",
                     "| split | counts (by strata) |", "|---|---|"]
    for k, v in splits_summary.items():
        report_lines.append(f"| {k} | {dict(v)} |")

    report_lines += ["", "## Leakage check", ""]
    if leaks:
        report_lines.append(f"**FAIL** — {len(leaks)} groups appeared in multiple splits")
        for g, a, b in leaks[:20]:
            report_lines.append(f"- `{g}` in {a} and {b}")
    else:
        report_lines.append("PASS — no group key appeared in more than one split")

    if drop_summary:
        report_lines += ["", "## Dedup", "",
                         f"- before: {drop_summary['before']}",
                         f"- after:  {drop_summary['after']}",
                         f"- dropped: {drop_summary['before'] - drop_summary['after']}"]

    (out / "PREP_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n[ok] wrote {out / 'data.yaml'}")
    print(f"[ok] wrote {out / 'PREP_REPORT.md'}")
    print(json.dumps(splits_summary, indent=2))
    return 0 if not leaks else 1


if __name__ == "__main__":
    raise SystemExit(main())
