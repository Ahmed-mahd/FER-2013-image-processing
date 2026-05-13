# FER2013 — Project Plan

> Deep Learning pipeline for Facial Expression Recognition using FER2013 dataset.
> **Team:** Ahmed Mahdi & teammate | **Course:** AI/ML Lab | **Last updated: 2026-05-13**

---

## Stage Completion Summary

| Stage | Name | Status |
|-------|------|--------|
| 1 | Dataset Acquisition | ✅ COMPLETE |
| 2 | EDA & Quality Checks | ✅ COMPLETE |
| 3 | Preprocessing & Augmentation | ✅ COMPLETE |
| 4 | CNN from Scratch | ✅ COMPLETE — 66.27% test accuracy |
| 5 | Transfer Learning (EfficientNetB0) | ✅ COMPLETE — 63.22% test accuracy |
| 6 | Evaluation & Comparison | ✅ COMPLETE |
| 7 | Repo Cleanup & Structure | ✅ COMPLETE |
| 8 | Documentation (HTML Report) | ✅ COMPLETE — `documentation/report.html` |
| 9 | Streamlit Deployment (Bonus) | ✅ COMPLETE |

---

## Stage 1 — Dataset Acquisition `[x] COMPLETE`

- [x] FER2013 downloaded via Kaggle API
- [x] Structure: `data/train/<emotion>/` and `data/test/<emotion>/` (7 class folders)
- [x] Total: **28,709 training** + **7,178 test** images (48×48 grayscale JPEG)

**Class distribution (train):**

| Emotion | Count |
|---------|-------|
| Angry | 3,995 |
| Disgust | 436 ← severe minority |
| Fear | 4,097 |
| Happy | 7,215 ← dominant |
| Neutral | 4,965 |
| Sad | 4,830 |
| Surprise | 3,171 |

---

## Stage 2 — EDA & Quality Checks `[x] COMPLETE`

- [x] Quality check on 210-image stratified batch (30 per class)
- [x] All images: 48×48, grayscale, 100% usable, zero corrupt
- [x] Brightness analysis: Surprise brightest (avg 158.6), Angry/Neutral darkest (~125)
- [x] Saved to: `output/reports/stage2_eda/`

**Key findings:**
- Happy (7,215) vs Disgust (436) = **16.6× imbalance** — must address
- Zero corrupt images — dataset is clean

---

## Stage 3 — Preprocessing & Augmentation `[x] COMPLETE`

- [x] **Pipeline A** (CNN Scratch): Grayscale → Resize 48×48 → Normalize [0,1]
- [x] **Pipeline B** (Transfer Learning): RGB → Resize 224×224 → EfficientNet preprocess_input [-1,1]
- [x] **Disgust** augmented ×4 on disk (horizontal flip, rotation ±20°, zoom ±15%)
- [x] **Class weights** computed via `sklearn.utils.class_weight.compute_class_weight`
- [x] Class weights saved to: `output/reports/stage3_preprocessing/class_weights.json`

---

## Stage 4 — CNN from Scratch `[x] COMPLETE`

- [x] Architecture: 4× Conv blocks + GlobalAveragePooling + Dense head
- [x] Optimizer: Adam (lr=1e-3) with ReduceLROnPlateau
- [x] Callbacks: ModelCheckpoint, EarlyStopping(patience=12), CSVLogger
- [x] Class weights applied during training
- [x] **60 epochs** (early stopping may terminate earlier)
- [x] Model saved: `models/cnn_scratch_best.keras`
- [x] Outputs: `output/reports/stage4_cnn_scratch/`

**Results:**
- Best Val Accuracy: **65.83%**
- **Test Accuracy: 66.27%** ← baseline
- Training Time: ~102 min (CPU)
- Parameters: 2,489,383

---

## Stage 5 — Transfer Learning (EfficientNetB0) `[x] COMPLETE`

**Why EfficientNetB0 over MobileNetV2:**
MobileNetV2 tested but peaked at 64% (couldn't break CNN scratch baseline of 66.3%).
EfficientNetB0 uses compound scaling (width+depth+resolution) → better fine-grained feature maps.

- [x] `src/models/transfer_learning.py` — EfficientNetB0 backbone (`include_top=False`, `weights='imagenet'`)
- [x] Custom classification head:
  - `GlobalAveragePooling2D()`
  - `Dense(256, relu)` + BatchNormalization + `Dropout(0.5)`
  - `Dense(128, relu)` + `Dropout(0.3)`
  - `Dense(7, softmax)`
- [x] Phase 1: LR=1e-3, 25 epochs (head only, base frozen)
- [x] Phase 2: LR=1e-5, 30 epochs (unfreeze last 50 layers, BN frozen)
- [x] tf.data pipeline with AUTOTUNE prefetch → GPU utilization ~50-60% vs ~25% with ImageDataGenerator
- [x] Model saved: `models/transfer_learning_best.keras`
- [x] Outputs: `output/reports/stage5_transfer_learning/`

**Results:**
- Best Val Accuracy: **65.5%** (epoch 55)
- **Test Accuracy: 63.22%**
- Training Time: **39.1 min** (GPU — tf.data pipeline)

---

## Stage 6 — Evaluation `[x] COMPLETE`

- [x] `src/evaluator.py` — evaluates both models on held-out test set (7,178 images)
- [x] Confusion matrix (dark-themed heatmap) — both models
- [x] ROC/AUC curves (one-vs-rest, all 7 emotions) — both models
- [x] Side-by-side model comparison bar chart
- [x] Outputs: `output/reports/stage6_evaluation/`

**Final Comparison:**

| Metric | CNN Scratch | EfficientNetB0 |
|--------|:-----------:|:--------------:|
| Test Accuracy | **66.23%** | 63.22% |
| Macro Precision | 68.05% | 62.68% |
| Macro Recall | 58.87% | 60.41% |
| Macro F1-Score | 59.77% | **60.80%** |
| Training Time | ~102 min | **39.1 min** |
| Parameters | 2.49M | 4.4M total |

**Per-class highlights (CNN Scratch):**
- Best: Happy (F1=0.87), Surprise (F1=0.78)
- Worst: Disgust (F1=0.32), Fear (F1=0.43)

---

## Stage 7 — Repository Cleanup `[x] COMPLETE`

- [x] `output/reports/` reorganized into stage subfolders:
  - `stage2_eda/` — EDA charts, quality report
  - `stage3_preprocessing/` — augmentation samples, class weights
  - `stage4_cnn_scratch/` — CNN training outputs
  - `stage5_transfer_learning/` — TL training outputs
  - `stage6_evaluation/` — confusion matrices, ROC curves, comparison
- [x] All script path references updated (`train_scratch.py`, `train_transfer.py`, `analytics.py`, `prepare_data.py`)
- [x] README.md rewritten with complete run instructions
- [x] MD files moved to `documentation/` folder

---

## Stage 8 — Documentation `[x] COMPLETE`

- [x] Full HTML report generated: `documentation/report.html`
  - Printable to PDF via browser Ctrl+P
  - Dataset description and task definition
  - Preprocessing steps with justification
  - Architecture of each model with layer tables
  - Evaluation results with confusion matrix screenshots
  - Comparison table and ROC curve screenshots
  - Conclusion and recommendations

---

## Stage 9 — Deployment (Bonus +1 to +4) `[x] COMPLETE`

- [x] `app.py` — Streamlit web application:
  - Upload any face image (jpg, png, webp)
  - Model selector: CNN Scratch or EfficientNetB0
  - Applies correct preprocessing pipeline per model
  - Shows predicted emotion with emoji + confidence %
  - Full 7-class confidence bar chart
  - Before/after CLAHE visualization (CNN mode)
  - Dark-themed premium UI

**Run:**
```bash
streamlit run app.py
# → http://localhost:8501
```

---

## Grading Checklist

| Component | Marks | Status |
|-----------|-------|--------|
| Full Project (Code + Dataset + Models + Evaluation) | 10 | ✅ Complete |
| Presentation & Documentation | 5 | ✅ `documentation/report.html` |
| Discussion / Viva | 5 | 🕐 Upcoming Week 14 |
| Deployment (Bonus) | +1 to +4 | ✅ Streamlit app running |

---

## Key Constraints (Satisfied)

> - ✅ No built-in datasets — FER2013 raw files from Kaggle
> - ✅ No drag-and-drop Transfer Learning — backbone built manually layer by layer
> - ✅ No pre-processed datasets — raw `.jpg` files, processed from scratch
> - ✅ Class imbalance addressed: class weights + Disgust ×4 augmentation
> - ✅ Separate evaluation pipeline for Generator vs tf.data

---

## Key Decisions Log

| # | Decision | Choice |
|---|----------|--------|
| 1 | Image size | **48×48 grayscale** (CNN) · **224×224 RGB** (TL) |
| 2 | TL backbone | **EfficientNetB0** (MobileNetV2 peaked at 64%, below baseline) |
| 3 | Imbalance fix | **Class weights + Disgust augmentation ×4** |
| 4 | Data pipeline | **tf.data + AUTOTUNE prefetch** → 8× faster than ImageDataGenerator |
| 5 | Evaluation | Separate predict functions per pipeline type |
| 6 | Deployment | Streamlit `app.py` — supports both models |

---

*Last updated: 2026-05-13 | All stages complete.*
