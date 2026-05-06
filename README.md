# METI Custom Vision — fire/smoke (YOLOv11)

Supplementary fire/smoke detection for the BANDARIYAH / Protex demo. YOLOv11 inference
service that emits webhooks into a small FastAPI demo platform with an SSE dashboard,
plus a multi-source dataset pipeline and training notebooks for the 4090 laptop.

## Two-machine setup

| Machine | Role | Install |
|---|---|---|
| Code-only (no NVIDIA) | Build and verify wiring | `pip install -r requirements.txt` |
| 4090 laptop | Datasets, training, demo | `pip install -r requirements.txt && pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu121` |

## Quickstart

Verify the platform/dashboard wiring without a model:

```bash
# Terminal 1
uvicorn service.demo_platform:app --port 8001
# Terminal 2
python scripts/verify_pipeline.py
# open http://localhost:8001/
```

Run the demo with a trained model on the 4090 laptop:

```bash
python service/inference_service.py
```

## Documentation

- [`docs/01-local-setup.md`](docs/01-local-setup.md) — installation on both machines
- [`docs/02-training-guide.md`](docs/02-training-guide.md) — training on the 4090 laptop
- [`docs/03-dataset-guide.md`](docs/03-dataset-guide.md) — multi-source dataset pipeline
- [`docs/04-drop-in-trained-model.md`](docs/04-drop-in-trained-model.md) — wiring `best.pt` into the demo
- [`docs/05-demo-runbook.md`](docs/05-demo-runbook.md) — day-of demo script

## Notebooks (4090 laptop)

- [`notebooks/01-explore-datasets.ipynb`](notebooks/01-explore-datasets.ipynb) — sample grids and class counts per raw source
- [`notebooks/02-prepare-merged-dataset.ipynb`](notebooks/02-prepare-merged-dataset.ipynb) — merge + split with diagnostics
- [`notebooks/03-train-yolo11.ipynb`](notebooks/03-train-yolo11.ipynb) — primary training entry point (Colab-portable)
- [`notebooks/04-evaluate-model.ipynb`](notebooks/04-evaluate-model.ipynb) — confusion matrix + error gallery

## Layout

```
service/         FastAPI receiver + SSE dashboard + YOLO inference loop + RTLS mock
scripts/         CLI tools: smoke test, pipeline verify, dataset download/prepare/inspect
notebooks/       Jupyter equivalents for the dataset and training stages
docs/            Per-stage documentation
data/            sources.yaml + raw/ + merged/ (the latter two gitignored)
models/          best.pt drops here after training (gitignored)
demo-clips/      Industrial fire footage for the demo (gitignored)
```
