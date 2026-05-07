# FER2013 — Facial Expression Recognition
**Phase 1 Complete | CNN from Scratch: 66.3% Test Accuracy**

---

## Documentation

| File | Purpose |
|------|--------|
| [README.md](README.md) | Setup and quick-start guide |
| [WALKTHROUGH.md](WALKTHROUGH.md) | Full technical walkthrough — Stages 1 to 4 |
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | Stage tracker and decisions log |
| [analytic_report.md](analytic_report.md) | Phase 1 dataset analysis report |

---

## Project Overview

This repository contains Phase 1 of a Facial Expression Recognition (FER) project using the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).

The goal of Phase 1 is to:
- Select a stratified random batch from the FER2013 training set
- Validate image quality (corrupt, blank, too dark/bright)
- Run a preprocessing pipeline (CLAHE, denoising, normalization)
- Generate visual and statistical analysis reports

## Dataset

**FER2013** — 48x48 grayscale facial expression images

| Emotion  | Training Images |
|----------|----------------|
| Angry    | 3,995          |
| Disgust  | 436            |
| Fear     | 4,097          |
| Happy    | 7,215          |
| Neutral  | 4,965          |
| Sad      | 4,830          |
| Surprise | 3,171          |
| **Total**| **28,709**     |

> The dataset is **not included** in this repo (too large). Download it from Kaggle — see Setup below.

---

## Project Structure

```
project/
├── src/
│   ├── batch_selector.py    # Stratified random batch selection
│   ├── quality_checker.py   # Image quality validation
│   ├── preprocessor.py      # CLAHE + blur + normalization pipeline
│   └── visualizer.py        # Analysis plot generation
├── output/
│   └── reports/             # Generated plots and quality CSV
├── main.py                  # Run Phase 1 pipeline
├── analytics.py             # Generate analytic dashboard
├── analytic_report.md       # Full Phase 1 report
└── requirements.txt
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Ahmed-mahd/FER-2013-image-processing.git
cd FER-2013-image-processing
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Configure your Kaggle credentials:
```bash
# Create ~/.kaggle/kaggle.json with your username and API key
# {"username": "your-username", "key": "your-api-key"}
```

Download and extract:
```bash
kaggle datasets download -d msambare/fer2013 -p data/ --unzip
```

### 4. Run Phase 1
```bash
# Default: 210 images (30/class), seed=42
python main.py

# Custom batch size and seed
python main.py --batch-size 350 --seed 99
```

### 5. Generate analytic report
```bash
python analytics.py
```

---

## Phase 1 Results

| Metric | Value |
|--------|-------|
| Batch size | 210 images (30/class) |
| Usable images | 210/210 (100%) |
| Preprocessing output | (210, 48, 48) float32 |
| Pixel mean (post-CLAHE) | 0.541 |
| Pixel std (post-CLAHE) | 0.206 |

See [analytic_report.md](analytic_report.md) for the full report with visualizations.

---

## Preprocessing Pipeline

```
Grayscale Load -> Resize 48x48 -> CLAHE -> Gaussian Blur -> Normalize [0,1]
```

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): enhances local contrast without amplifying noise
- **Gaussian Blur** (3x3): light denoising
- **Normalization**: scales pixel values to [0.0, 1.0] float32 for model input

---

## Stages Completed

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Dataset download & project setup | Done |
| 2 | Batch sampling, quality checks, preprocessing | Done |
| 3 | Full preprocessing pipeline (augmentation, split, class weights) | Done |
| **4** | **CNN from Scratch — 66.3% test accuracy** | **Done** |
| 5 | Transfer Learning (MobileNetV2) | Next |
| 6 | Evaluation (confusion matrix, F1, ROC) | Pending |
| 7 | Repo cleanup | Pending |
| 8 | Report & slides | Pending |
| 9 | Deployment — Streamlit (Bonus) | Optional |

---

## Phase 1 Results

| Metric | Value |
|--------|-------|
| Full training set | 28,709 images |
| Batch validated | 210/210 usable (100%) |
| Preprocessing | Grayscale, CLAHE, blur, normalize [0,1] |
| Disgust augmentation | 436 → 2,180 (+1,744 images) |
| Train / Val / Test split | 25,888 / 4,565 / 7,178 |

## CNN from Scratch Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **66.27%** |
| Best Val Accuracy | 65.83% (epoch 57) |
| Total Parameters | 2,489,383 |
| Training Time | ~102 min (CPU) |
| Architecture | 4 Conv Blocks + Dense Head + Softmax(7) |

See [WALKTHROUGH.md](WALKTHROUGH.md) for full details.

---

## Phases

- [x] **Phase 1** — Batch selection, quality checks, preprocessing, analysis
- [x] **Phase 2 (Stage 4)** — CNN from scratch trained and evaluated
- [ ] Phase 3 — Transfer Learning (MobileNetV2)
- [ ] Phase 4 — Evaluation & model comparison
- [ ] Phase 5 — Report, slides, deployment
