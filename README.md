# FER2013 — Facial Expression Recognition
**Phase 1: Dataset Exploration & Preprocessing**

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

## Phases

- [x] **Phase 1** — Batch selection, quality checks, preprocessing, analysis
- [ ] Phase 2 — Model architecture & training
- [ ] Phase 3 — Evaluation & optimization
