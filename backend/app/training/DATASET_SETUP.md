# 📋 HAM10000 / ISIC Dataset Setup & Licence Guide

## ⚖️ Licence & Attribution (Mandatory CC BY-NC 4.0)

The **HAM10000** ("Human Against Machine with 10,000 training images") dataset is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** licence.

### Licence Restrictions:
- **Non-Commercial Use Only:** You may not use this material or models trained on this dataset for commercial purposes without explicit permission.
- **Attribution:** Any research, publication, or deployment must cite the original authors:
  > Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci. Data* 5, 180161 (2018). [https://doi.org/10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

---

## 💾 Storage & Hardware Requirements

| Resource | Minimum Requirement | Recommended | Local System Status |
| :--- | :--- | :--- | :--- |
| **Disk Space** | 12 GB free (5.3 GB download + 5.6 GB extracted) | 20 GB free | **113.99 GB Free (Drive C:)** ✅ |
| **System RAM** | 8 GB | 16 GB | **15.77 GB RAM** ✅ |
| **GPU / Acceleration** | CUDA GPU or modern multi-core CPU | NVIDIA RTX 6GB+ VRAM | **NVIDIA GeForce MX230 (2GB VRAM)** / CPU fallback ✅ |
| **Driver & CUDA** | Driver 450+ / CUDA 11+ | Driver 525+ / CUDA 12+ | **NVIDIA Driver 581.95 / CUDA 13.0** ✅ |
| **Python** | 3.10+ | 3.11 - 3.14 | **Python 3.14.7 (System) / 3.13.15 (.venv)** ✅ |

---

## 📦 Exact Download Sizes

| Component | Format | Size | Description |
| :--- | :--- | :--- | :--- |
| **HAM10000_metadata.tab / .csv** | TAB or CSV | ~2.5 MB | Diagnostic labels, lesion_id, age, sex, anatomic site |
| **HAM10000_images_part_1.zip** | ZIP Archive | ~2.7 GB | First 5,000 dermoscopy images (ISIC_0024306 to ISIC_0029305) |
| **HAM10000_images_part_2.zip** | ZIP Archive | ~2.6 GB | Second 5,015 dermoscopy images (ISIC_0029306 to ISIC_0034320) |
| **Total Dataset Download** | **Compressed** | **~5.3 GB** | **Extracted size: ~5.6 GB (10,015 images total)** |
| **PyTorch + CUDA 12.1 Wheels** | Python wheels | ~2.5 GB | `torch`, `torchvision`, `onnx`, `onnxruntime` |

---

## 📥 Dataset Acquisition Options

### Option A: Official Harvard Dataverse (Primary Source)
- **URL:** [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Download the following files:
  1. `HAM10000_metadata.csv` (2.5 MB)
  2. `HAM10000_images_part_1.zip` (2.7 GB)
  3. `HAM10000_images_part_2.zip` (2.6 GB)

### Option B: Kaggle CLI (Automated Single-Command)
If you have a Kaggle account and API token (`kaggle.json` placed in `%USERPROFILE%\.kaggle\`):
```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ham10000 --unzip
```

### Option C: ISIC Archive Official Portal
- **URL:** [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)
- Select **ISIC 2018 Task 3: Disease Classification** (Training Data: 10,015 images).

---

## 📁 Required Local Directory Layout

Place or extract the downloaded files inside the project under `data/ham10000/` or on an external drive (e.g. `E:\DermoraData\ham10000\`):

```
<DATA_ROOT>/
├── HAM10000_metadata.tab (or HAM10000_metadata.csv)
├── HAM10000_images_part_1/
│   ├── ISIC_0024306.jpg
│   └── ... (5,000 images)
└── HAM10000_images_part_2/
    ├── ISIC_0029316.jpg
    └── ... (5,015 images)
```

The pipeline automatically detects:
- Both `HAM10000_metadata.tab` (Harvard Dataverse) and `HAM10000_metadata.csv` (Kaggle).
- Both image folders (`HAM10000_images_part_1` and `HAM10000_images_part_2`).
- External drive paths via `--data-dir "E:\DermoraData\ham10000"`, avoiding any file copies into OneDrive.

---

## 🧪 Pipeline Self-Verification Test (Runs Instantly)

You can verify the entire data splitting, leakage check, class balancing, and image triage pipeline right now without downloading the dataset:

```bash
python backend/app/training/verify_pipeline.py
```

---

## 🚀 PyTorch Dependencies Installation

When ready to install training dependencies (verified for Python 3.13 on Windows):

```bash
# For NVIDIA CUDA GPU acceleration (Driver 581.95 / CUDA 12.4+):
.\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For ONNX model export and parity testing:
.\.venv\Scripts\pip.exe install onnx onnxruntime
```

---

## 🏃 Running the Training Pipeline

### Recommended Settings for Local NVIDIA MX230 (2GB VRAM):
Because the NVIDIA GeForce MX230 has 2048 MB VRAM, the pipeline automatically employs **Automatic Mixed Precision (AMP)** and memory cache clearing. Use mini-batch size 8 with gradient accumulation steps 2 (giving an effective batch size of 16):

```bash
# EfficientNet-B0 (Recommended: fast, high accuracy, low VRAM footprint):
python -m backend.app.training.train --data-dir data/ham10000 --backbone efficientnet_b0 --loss weighted_ce --epochs 25 --batch-size 8 --accum-steps 2

# ResNet-34 alternative:
python -m backend.app.training.train --data-dir data/ham10000 --backbone resnet34 --loss weighted_ce --epochs 25 --batch-size 8 --accum-steps 2

# CPU Execution (if CUDA is not preferred):
python -m backend.app.training.train --data-dir data/ham10000 --backbone efficientnet_b0 --device cpu --epochs 15 --batch-size 16
```

### Dry-Run / Smoke Test Mode:
To test the full training loop, backpropagation, checkpoint saving, test evaluation report generation, and ONNX export on synthetic tensors prior to downloading the 5.3 GB dataset:
```bash
python -m backend.app.training.train --dry-run
```

### Safe Checkpoint Resumption:
To resume training from an existing checkpoint at `epoch + 1`:
```bash
python -m backend.app.training.train --data-dir "E:\DermoraData\ham10000" --resume backend/app/training/artifacts_real_training_20261020/latest_model.pt --epochs 25 --batch-size 8 --accum-steps 2
```
- Reuses existing `train_split.csv`, `val_split.csv`, and `test_split.csv` to ensure 0% partition drift.
- Appends to existing `training_log.txt` without truncating prior epoch logs.
- Restores model weights, optimizer momentum, AMP GradScaler state, scheduler, and early-stopping counters.
- For legacy checkpoints lacking optimizer/scheduler states, explicitly warns that exact-state resume is unavailable and performs a weights-only warm start.
- Checkpoints are written atomically to temporary swap files before atomic replacement, guaranteeing corruption protection.

The pipeline will automatically:
1. Verify metadata and image integrity.
2. Partition the dataset with **stratified lesion_id grouping** (0% patient/lesion leakage).
3. Compute inverse-frequency class weights to handle severe class imbalance.
4. Fine-tune pre-trained ImageNet weights with Cosine Annealing learning rate schedule.
5. Save the best checkpoint (`best_model.pt`) based on validation Balanced Multi-class Accuracy.
6. Run independent test evaluation on the unseen partition and save `test_evaluation_report.json`.
7. Export the trained model to `dermora_model.onnx` and verify numerical parity.

