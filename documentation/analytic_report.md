# FER2013 — Phase 1 Analytic Report

**Real Kaggle Dataset | Batch: 210 images | Seed: 42**

> *Note: All visualizations reference `output/reports/stage2_eda/` for EDA outputs and `output/reports/stage3_preprocessing/` for preprocessing outputs.*

---

## Analytics Dashboard

![Analytics Dashboard](../output/reports/stage2_eda/analytics_dashboard.png)

---

## 1. Dataset Overview

| Metric | Value |
|--------|-------|
| Full training set | **28,709 images** |
| Emotion classes | **7** |
| Batch selected | **210 images** |
| Sampling ratio | **0.73% of training set** |
| Usable images | **210 / 210 (100%)** |
| Flagged / corrupt | **0** |
| Batch per class | **30 images (perfectly stratified)** |

---

## 2. Full Dataset Class Imbalance

| Emotion  | Full Dataset | Batch Sampled | Sample %  |
|----------|-------------|--------------|-----------|
| Angry    | 3,995       | 30           | 0.75%     |
| Disgust  | **436**     | 30           | **6.88%** |
| Fear     | 4,097       | 30           | 0.73%     |
| Happy    | **7,215**   | 30           | 0.42%     |
| Neutral  | 4,965       | 30           | 0.60%     |
| Sad      | 4,830       | 30           | 0.62%     |
| Surprise | 3,171       | 30           | 0.95%     |

> **Warning — Critical class imbalance:** `Happy` (7,215) has **16.6x more samples** than `Disgust` (436).
> Addressed in Stage 3 via class weighting + disk augmentation (Disgust ×4).

---

## 3. Brightness Statistics per Class

| Emotion  | Avg Brightness | Avg Contrast (Std) | Observation                          |
|----------|---------------|-------------------|--------------------------------------|
| Angry    | 126.2         | 52.4              | Slightly darker faces                |
| Disgust  | 136.4         | 59.4              | Higher contrast                      |
| Fear     | 130.9         | **59.9**          | Highest contrast — most variation    |
| Happy    | 129.5         | 49.6              | Consistent, lower contrast           |
| Neutral  | 124.9         | 57.5              | Darker on average                    |
| Sad      | 124.9         | 51.4              | Similar to neutral                   |
| Surprise | **158.6**     | 51.2              | Brightest class — open eyes/mouth    |

**Batch-wide brightness:**
- Range: `34.5 - 219.4`
- Mean:  `133.1`
- Std:   `34.4` — healthy spread, no cluster near extremes

---

## 4. Image Quality Validation

| Check                   | Result       |
|-------------------------|--------------|
| All images loadable     | Yes          |
| All images 48 x 48 px   | Yes          |
| All images grayscale    | Yes          |
| Corrupt files           | None         |
| Too-dark images (< 10)  | None         |
| Too-bright images (>245)| None         |
| Blank/uniform images    | None         |
| **Overall usability**   | **100%**     |

---

## 5. Preprocessing Validation

After the full pipeline (resize + normalize — no CLAHE in training pipeline):

| Metric           | Value             |
|------------------|-------------------|
| Output dtype     | `float32`         |
| Output shape     | `(210, 48, 48)`   |
| Pixel range      | `[0.0, 1.000]`    |
| Mean pixel value | `0.541`           |
| Std pixel value  | `0.206`           |

> CLAHE is applied for exploratory visualization only — the training pipeline uses plain rescale [0,1].

---

## 6. Sample Visualizations

### Raw Samples (Before Preprocessing)
![Raw Samples](../output/reports/stage2_eda/sample_grid_raw.png)

### Before vs After (CLAHE — visual reference only)
![Before After](../output/reports/stage3_preprocessing/before_after_comparison.png)

### Pixel Intensity Histograms
![Pixel Histograms](../output/reports/stage3_preprocessing/pixel_histograms.png)

### Brightness Distribution per Class
![Brightness Distribution](../output/reports/stage2_eda/brightness_distribution.png)

---

## 7. Key Findings & Recommendations

| # | Finding | Recommendation |
|---|---------|----------------|
| 1 | Disgust severely underrepresented (436 vs 7,215 Happy) | Augmentation ×4 + class weights ✅ |
| 2 | Surprise is the brightest class (avg 158.6) | Monitor brightness normalization effects |
| 3 | Fear & Disgust have highest contrast (~59 std) | Most textured faces — good discriminative features |
| 4 | Zero corrupt or unusable images | Dataset is clean — no filtering needed ✅ |
| 5 | Stratified batch ensures fair class evaluation | Good baseline for analysis ✅ |
