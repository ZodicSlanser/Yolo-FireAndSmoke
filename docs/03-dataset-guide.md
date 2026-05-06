# 03 — Multi-source dataset guide

The "more the merrier" principle is real for fire/smoke. Each source has its own systematic biases — D-Fire is heavy on outdoor wildfire, Roboflow community projects skew toward dramatic visible flame, FASDD has more industrial / surveillance footage. Merging across 4–7 sources gives the model variety it can't get from one.

The whole pipeline is driven by `data/sources.yaml`. Adding a new dataset = adding an entry. No code changes.

---

## Pipeline overview

```
data/sources.yaml
        │
        ▼  (auth: ROBOFLOW_API_KEY, kaggle.json, hf token)
scripts/download_datasets.py  ──►  data/raw/<source>/
        │
        ▼
notebooks/01-explore-datasets.ipynb   (sanity-check sources before merging)
        │
        ▼
scripts/prepare_dataset.py    ──►  data/merged/{train,val,test}/
                                   data/merged/data.yaml
                                   data/merged/PREP_REPORT.md
        │
        ▼
scripts/inspect_dataset.py    ──►  sample grids + balance stats
```

---

## `data/sources.yaml` schema

```yaml
sources:
  - name: dfire             # used as filename prefix and target dir name
    type: git               # one of: git, roboflow, huggingface, kaggle, url, local
    url: https://github.com/gaiasd/DFireDataset.git
    layout: yolo            # informational; the prepare script auto-detects layout
    class_map: { 0: fire, 1: smoke }   # SOURCE-INDEX → unified-name
    weight: 1.0             # train-set downsample factor; 1.0 keeps all
```

`class_map` is the most important field. It translates each source's class indices into the unified vocabulary `[fire, smoke]`. The prepare script then reverse-maps unified names to indices `[0=fire, 1=smoke]`. Sources lacking one of the two classes (e.g. fire-only) are merged in cleanly — their label files just won't contain class `1`.

`weight` controls **train-set** downsampling. Val and test splits are never downsampled — we want honest evaluation. A source with `weight=0.5` gets ~50% of its train images randomly kept.

### Source types

- **`git`** — clones the repo. Use for D-Fire and similar GitHub-hosted YOLO datasets.
- **`roboflow`** — uses the Roboflow Python SDK. Requires `ROBOFLOW_API_KEY` env var. Specify `workspace`, `project`, `version`.
- **`kaggle`** — shells out to the Kaggle CLI. Requires `~/.kaggle/kaggle.json`. Specify `handle` (e.g. `dataclusterlabs/fire-and-smoke-dataset`).
- **`huggingface`** — uses `huggingface_hub.snapshot_download`. Specify `repo`. Public repos work without auth.
- **`url`** — plain HTTP zip download + extract. Specify `url`. No auth.
- **`local`** — already-on-disk source under `data/raw/<name>/` or `path:`. Use for site-collected industrial negative footage.

---

## Recommended source list

The default `data/sources.yaml` ships with these sources. Add more freely.

| Source | Type | Approx size | What it adds |
|---|---|---:|---|
| `dfire` | git | ~21k | Academic standard. Outdoor + indoor mix. Both classes. |
| `rf_primary` | roboflow | ~10k | Crowd-sourced. Both classes. |
| `rf_continuous_fire` | roboflow | ~5k | Fire-only. Adds visible-flame variety. |
| `rf_smoke_detection` | roboflow | ~3k | Smoke-only. Helps balance the smoke class. |
| `kaggle_firesmoke` | kaggle | ~7k | Different annotator style; reduces label bias. |
| `industrial_negatives` | local | varies | Site-collected steam/dust/welding negatives. |

Other good sources to consider adding:

- **FASDD** (Flame and Smoke Detection Dataset) — ~95k images, several research labs host it on HF and zenodo. Highest variety per image but you should weight it down (`weight: 0.3`) to avoid dominance.
- Additional Roboflow community fire/smoke projects — search `universe.roboflow.com`. Diminishing returns past 5–6 sources.
- Synthetic data from GAN-generated fire/smoke compositions — useful for the negative-class step in `02-training-guide.md` but not for the main training set.

---

## Merge + split details

`scripts/prepare_dataset.py` runs five stages.

### 1. Normalize labels

For each source, walks image+label pairs, translates class indices via `class_map`, writes to `data/staging/<source>/{images,labels}/`. Filenames prefixed (`dfire_xxx.jpg`) to prevent collisions.

The walker is layout-tolerant — it accepts:

- `<root>/{train,valid,val,test}/images/*.jpg` + `.../labels/*.txt`
- `<root>/images/*.jpg` + `<root>/labels/*.txt`
- `<root>/*.jpg` + `<root>/*.txt`

### 2. Deduplicate

Computes a 64-bit perceptual hash (`imagehash.phash`) for every image. Two-pass:

- **Within-source** dedup — handles compression-artifact duplicates (Hamming distance ≤ 4).
- **Cross-source** dedup — D-Fire and some Roboflow sets share a few hundred frames. Same threshold.

~30 s per 10k images on the 4090 laptop. Skip with `--no-dedup` if you're iterating.

### 3. Group-aware stratified split

This is the part most pipelines get wrong. Two principles:

- **Group key** = source name + filename prefix before `_frame_<N>` if present. Frames `clip_007_frame_010.jpg` and `clip_007_frame_011.jpg` MUST land in the same split — they're the same physical scene. Without grouping, you leak the test set into training and the eval is fake.
- **Stratify by class presence**: each image is tagged `fire_only` / `smoke_only` / `both` / `negative`. The split keeps these proportions even across train/val/test. Without stratification, you can get a smoke-poor val set and the validation mAP becomes unstable.

Implementation: `sklearn.model_selection.GroupShuffleSplit` twice — first peel off test, then val from the remainder.

Default ratios: 80/10/10. Deterministic seed 42 written into `data.yaml`.

### 4. Source-weight downsample

Train-only. Each source's images get kept with probability `weight`. Val and test are untouched.

This is how you keep large sources from dominating without throwing away their eval value. A 95k-image dataset at `weight: 0.3` contributes ~28k train images and 100% of its val/test.

### 5. Write

Final layout:

```
data/merged/
├── data.yaml
├── train/{images,labels}/
├── val/{images,labels}/
├── test/{images,labels}/
└── PREP_REPORT.md
```

The report has:

- Per-source normalization counts (images, fire labels, smoke labels, negatives)
- Per-split strata counts
- Dedup before/after counts
- **Leakage check** — confirms no group key appears in more than one split

If the leakage check fails, the prepare script exits non-zero. Fix it before training.

---

## Fast iteration

```bash
# Skip dedup and weights for a quick first run (~60 s on 30k images)
python scripts/prepare_dataset.py --no-dedup --no-weights --train 0.9 --val 0.1 --test 0.0
```

Then once you have a baseline number to beat, run the full pipeline.

---

## Adding a new dataset

1. Append an entry to `data/sources.yaml` with the right `type` and `class_map`.
2. `python scripts/download_datasets.py --only your_new_source`
3. Open `notebooks/01-explore-datasets.ipynb` and look at the new source's sample grid + class counts. **If the class indices don't match what you wrote in `class_map`, fix `class_map` before continuing.** Common gotcha: Roboflow versions sometimes swap fire/smoke indices between v1 and v2 of the same project.
4. `python scripts/prepare_dataset.py` to re-merge.
5. Compare class balance in `PREP_REPORT.md` to the previous run. If smoke went down a lot, the new source is fire-heavy and you may want to weight it down.
