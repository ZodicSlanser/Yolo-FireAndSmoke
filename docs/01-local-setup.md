# 01 — Local setup

This repo runs on two kinds of machine:

- **Code-only machine** (no NVIDIA GPU): build and verify the platform/dashboard wiring. CPU PyTorch.
- **4090 laptop**: the real workflow — download datasets, train, evaluate, run the demo. CUDA PyTorch.

Same source tree on both. Only the requirements files differ.

---

## On either machine

```bash
cd "D:/Work/METI/PoCs/Yolo-FireAndSmoke"
python -m venv .venv

# bash on Windows:
source .venv/Scripts/activate
# OR PowerShell:
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

That installs CPU PyTorch + Ultralytics + FastAPI + dataset-prep tools. Verify:

```bash
python -c "import torch, cv2, ultralytics, fastapi, sklearn, imagehash; print('ok')"
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

If `cuda:` prints `False`, you're on CPU. That's expected for the code-only machine.

## Smoke test

```bash
python scripts/smoke_test.py
```

Downloads `yolo11n.pt` (first run only) and runs it on Ultralytics' bus.jpg. Prints detected classes and saves an annotated image under `runs/smoke_test/`. If this fails, nothing else will work — fix Ultralytics install first.

## Verify the demo platform end-to-end (no model required)

```bash
# Terminal 1
uvicorn service.demo_platform:app --port 8001
# Terminal 2
python scripts/verify_pipeline.py
```

Open http://localhost:8001/ in a browser — you should see fire/smoke cards arrive.

---

## On the 4090 laptop only

After `requirements.txt`, upgrade torch to the CUDA build:

```bash
pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True NVIDIA GeForce RTX 4090 Laptop GPU`. If it prints `False` after the CUDA install, your driver is old — install the latest NVIDIA Studio or Game Ready driver.

### One-time setup checklist for training

- [ ] Latest NVIDIA driver (Studio recommended for training)
- [ ] Plugged into wall power
- [ ] Power profile: "Best Performance"
- [ ] Cooling: hard flat surface or pad — laptop 4090s thermal-throttle on a couch
- [ ] 80+ GB free disk (the merged dataset can hit 30 GB)
- [ ] `nvidia-smi -l 5` ready in a side terminal during training to watch temps and VRAM

### Auth for dataset sources

`scripts/download_datasets.py` reads `data/sources.yaml`. Each source type has its own auth path:

- **Roboflow**: get a free API key from roboflow.com → Settings → API. Set `ROBOFLOW_API_KEY=...` in your shell.
- **Kaggle**: download `kaggle.json` from kaggle.com → Account → Create New Token. Save to `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`).
- **Hugging Face**: `pip install huggingface_hub && huggingface-cli login` (only needed for gated repos).
- **Git / URL / local**: no auth.
