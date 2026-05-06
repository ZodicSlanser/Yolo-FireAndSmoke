# 02 — Training guide (4090 laptop)

This is the document followed on the 4090 laptop after cloning the repo. Multi-source dataset acquisition is in [`03-dataset-guide.md`](03-dataset-guide.md); this doc is about training the model once the dataset is ready.

---

## Compute notes

You're training on an RTX 4090 mobile (16 GB VRAM). That's roughly 70–80% of a desktop 4090. For this workload, it's all upside.

Practical setup before training:

1. **Drivers & CUDA.** Latest NVIDIA driver. CUDA toolkit installation isn't strictly required — the CUDA-enabled PyTorch wheels bundle the runtime. Verify with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`.
2. **Power.** Wall power. Windows power plan = "Best Performance". Battery mode throttles the GPU 30–50%.
3. **Thermals.** Hard flat surface, cooling pad if you have one. `nvidia-smi -l 5` in a side terminal — sustained ≥80°C means you're throttling.
4. **Disk.** Merged dataset can hit 20–40 GB depending on which sources you pull. 80+ GB free.

Ballpark training-time targets with `cache=ram` and `amp=True`:

| Dataset size (merged) | `yolo11n` 80 epochs | `yolo11s` 80 epochs | `yolo11m` 80 epochs |
|---|---|---|---|
| 10 k images | 25–35 min | 45–60 min | 80–110 min |
| 30 k images | 60–90 min | 90–140 min | 3–4 hr |
| 60 k images | 2–2.5 hr | 3–4 hr | 6–8 hr |

If iterating on dataset choices, train `yolo11n` first (it's a dress rehearsal), then `yolo11s` once the data pipeline is settled.

---

## Stage 1 — baseline training

The notebook is the primary path: open [`notebooks/03-train-yolo11.ipynb`](../notebooks/03-train-yolo11.ipynb) in JupyterLab and run cells top-to-bottom. Same effect as the CLI command below.

```bash
yolo detect train \
  model=yolo11s.pt \
  data=data/merged/data.yaml \
  epochs=80 \
  imgsz=640 \
  batch=32 \
  device=0 \
  patience=20 \
  optimizer=AdamW \
  lr0=0.001 \
  weight_decay=0.0005 \
  warmup_epochs=3 \
  cos_lr=True \
  amp=True \
  cache=ram \
  project=runs/firesmoke \
  name=v11s-baseline \
  plots=True
```

Why each flag:

- `epochs=80, patience=20` — 80 cap, early-stop after 20 stagnant epochs. Fire/smoke usually converges around epoch 40–60.
- `imgsz=640` — standard. Fire/smoke is a "things, not stuff" detection problem at this scale.
- `batch=32` — fits comfortably on the 4090 laptop's 16 GB. Drop to 16 if you OOM.
- `optimizer=AdamW, lr0=0.001` — converges faster than SGD on small/medium datasets.
- `cos_lr=True, warmup_epochs=3` — modern best practice. Worth a couple mAP points.
- `amp=True` — mixed precision; ~30% faster, free.
- `cache=ram` — caches dataset in memory after epoch 1. Switch to `disk` if RAM-constrained.
- `plots=True` — generates PR curves, confusion matrix, F1 curves for B4.

Output lands in `runs/firesmoke/v11s-baseline/`. The thing you want is `weights/best.pt`.

---

## Stage 2 — validation gates

Open `runs/firesmoke/v11s-baseline/results.csv`, look at the final row. Compare against this table:

| Metric | Acceptable | Good | Concerning |
|---|---|---|---|
| `metrics/mAP50(B)` overall | ≥ 0.80 | ≥ 0.88 | < 0.75 |
| `metrics/mAP50(B)` fire | ≥ 0.85 | ≥ 0.92 | < 0.78 |
| `metrics/mAP50(B)` smoke | ≥ 0.70 | ≥ 0.82 | < 0.65 |
| `metrics/mAP50-95(B)` overall | ≥ 0.50 | ≥ 0.60 | < 0.42 |
| Train/val loss gap (last 10 epochs) | < 25% | < 15% | > 40% (overfit) |

Smoke is harder than fire — it's diffuse and looks like steam, dust, fog. A 10-point gap between fire and smoke mAP is normal. >25 points means smoke labels are inconsistent across the dataset (a known D-Fire issue).

Eyeball `confusion_matrix.png`. fire↔smoke confusion is fine (they co-occur). High `smoke→background` (false negatives) means the model misses real smoke — you need more positive smoke examples.

Then run [`notebooks/04-evaluate-model.ipynb`](../notebooks/04-evaluate-model.ipynb) for the **error gallery** — visual inspection of the worst false positives and false negatives. This is what tells you what kind of mistakes the model makes. Critical input for Stage 3.

---

## Stage 3 — industrial false-positive suppression

Out-of-the-box trained models trigger on:

- Sunset / sunlight on glass (→ "fire")
- Welding sparks (→ "fire")
- Steam vents (→ "smoke")
- Dust clouds from forklifts (→ "smoke")
- Orange high-vis vests in low light (→ "fire")
- Distant clouds through skylights (→ "smoke")

The fix is a **negative-class fine-tune**:

1. Collect 200–500 frames per false-positive category. Site footage is best; otherwise public industrial CCTV / training videos.
2. For each frame, create an **empty `.txt` label file** with the same name. Empty label = "no fire, no smoke." YOLO learns from this as a hard negative.
3. Drop these into `data/raw/industrial_negatives/{images,labels}/`. Re-run `scripts/prepare_dataset.py` to merge them into the training set.
4. Continue training **from your current best.pt**, not from scratch:

```bash
yolo detect train \
  model=runs/firesmoke/v11s-baseline/weights/best.pt \
  data=data/merged/data.yaml \
  epochs=20 \
  imgsz=640 \
  batch=32 \
  lr0=0.0001 \
  project=runs/firesmoke \
  name=v11s-neg-tuned
```

20 epochs at a 10× lower LR. Enough to teach the negatives without forgetting what was learned. Full LR will catastrophically forget — don't.

---

## Stage 4 — export

```bash
# ONNX — universal
yolo export model=runs/firesmoke/v11s-neg-tuned/weights/best.pt format=onnx opset=12 simplify=True

# TensorRT — for NVIDIA edge (Jetson Orin Nano), FP16
yolo export model=runs/firesmoke/v11s-neg-tuned/weights/best.pt format=engine half=True device=0

# OpenVINO — for Intel CPU/GPU edge
yolo export model=runs/firesmoke/v11s-neg-tuned/weights/best.pt format=openvino
```

Hand off all four (PT, ONNX, engine, OpenVINO) so the deployment target picks its runtime.

File sizes for `yolo11s`: PT ~22 MB · ONNX ~22 MB · TensorRT FP16 ~12 MB · OpenVINO ~22 MB folder.

---

## Common training failures

| Symptom | Likely cause | Fix |
|---|---|---|
| mAP plateaus low (0.5–0.6) by epoch 10 | Class index mismatch in label files after merge | Inspect `data.yaml` `names`; grep first int in each `.txt` |
| Train loss drops fast, val loss climbs after epoch ~15 | Overfit | More data, higher `weight_decay`, more aug, fewer epochs |
| `CUDA out of memory` mid-epoch | Batch too big | Halve `batch=`, optionally drop `imgsz=512` |
| `RuntimeError: stack expects each tensor to be equal size` | Corrupt image | `yolo detect val` surfaces bad files; delete and retry |
| Dataloader workers crash | Too many workers (Colab) | Add `workers=2` |
| Boxes predicted at center of image | Label-format bug | Verify YOLO `class cx cy w h` all in [0,1] |
