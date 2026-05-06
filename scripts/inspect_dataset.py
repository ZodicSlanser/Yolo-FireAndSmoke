"""Quick QA report on a merged dataset directory: counts, balance, sample grid."""
import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "merged"


def _count_split(split_dir: Path) -> dict:
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.exists():
        return {"missing": True}
    images = list(img_dir.iterdir())
    counts = Counter()
    boxes = Counter()
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists() or lbl.stat().st_size == 0:
            counts["negative"] += 1
            continue
        has = set()
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                if idx == 0:
                    has.add("fire")
                    boxes["fire"] += 1
                elif idx == 1:
                    has.add("smoke")
                    boxes["smoke"] += 1
        if has == {"fire"}:
            counts["fire_only"] += 1
        elif has == {"smoke"}:
            counts["smoke_only"] += 1
        elif has == {"fire", "smoke"}:
            counts["both"] += 1
        else:
            counts["negative"] += 1
    return {"images": len(images), "by_strata": dict(counts), "boxes": dict(boxes)}


def _sample_grid(split_dir: Path, out: Path, n: int = 16, seed: int = 42) -> None:
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.exists():
        return
    pool = list(img_dir.iterdir())
    if not pool:
        return
    random.Random(seed).shuffle(pool)
    pool = pool[:n]
    cols = 4
    rows = (len(pool) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else list(axes)
    for ax, p in zip(axes, pool):
        try:
            with Image.open(p) as im:
                ax.imshow(im)
        except Exception:
            ax.text(0.5, 0.5, "load fail", ha="center", va="center")
        lbl = lbl_dir / (p.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            with open(lbl) as f:
                indices = {int(line.split()[0]) for line in f
                           if line.strip() and line.split()[0].lstrip("-").isdigit()}
            tags = []
            if 0 in indices:
                tags.append("fire")
            if 1 in indices:
                tags.append("smoke")
            ax.set_title(",".join(tags) or "neg", fontsize=9)
        else:
            ax.set_title("neg", fontsize=9)
        ax.axis("off")
    for ax in axes[len(pool):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[ok] sample grid -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.dataset)
    if not (root / "data.yaml").exists():
        print(f"[fail] {root}/data.yaml not found — run prepare_dataset.py first")
        return 1

    print(f"[info] inspecting {root}")
    for split in ("train", "val", "test"):
        info = _count_split(root / split)
        if info.get("missing"):
            continue
        print(f"\n[{split}]")
        print(f"  images: {info['images']}")
        print(f"  by_strata: {info['by_strata']}")
        print(f"  boxes: {info['boxes']}")
        _sample_grid(root / split, root / f"sample_{split}.png", n=args.samples)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
