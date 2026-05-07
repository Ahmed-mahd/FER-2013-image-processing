# FER2013 Image Processing & Computer Vision — Project Plan

**Course:** Image Processing & Computer Vision  
**Dataset:** FER2013 — Facial Expression Recognition (28,709 images, 7 classes)  
**Repo:** https://github.com/Ahmed-mahd/FER-2013-image-processing  
**Task:** Multi-class Image Classification (7 emotions)

---

## Progress Legend
- `[x]` Completed
- `[/]` In progress
- `[ ]` Not started yet

---

## Stage 1 — Dataset & Project Setup `[x] COMPLETE`

- [x] Select a real dataset (≥1,000 images, no built-in library datasets)
  - FER2013: 28,709 training images, 7 emotion classes, raw .jpg files
- [x] Download dataset via Kaggle API (not pre-processed)
- [x] Verify raw file structure: `data/train/<emotion>/*.jpg`
- [x] Set up GitHub repository with proper structure
- [x] Write `README.md` and `requirements.txt`
- [x] Set up `.gitignore` (exclude `/data`, generated files)
- [x] Initial commit with project skeleton

---

## Stage 2 — Phase 1: Batch Sampling & Exploratory Analysis `[x] COMPLETE`

- [x] Build `src/batch_selector.py` — stratified random batch (210 images, 30/class)
- [x] Build `src/quality_checker.py` — detect corrupt, too-dark, too-bright, blank images
- [x] Run quality checks on batch → 210/210 usable (100%)
- [x] Save quality report CSV (`output/reports/quality_report.csv`)
- [x] Build `src/visualizer.py` — 6 analysis plots:
  - [x] Class distribution bar chart
  - [x] Brightness violin plot per class
  - [x] Raw sample grid (before preprocessing)
  - [x] Before vs After comparison (CLAHE effect)
  - [x] Pixel intensity histograms (post-preprocessing)
  - [x] Usability quality summary chart
- [x] Build 9-panel analytics dashboard (`analytics_dashboard.png`)
- [x] Write `analytic_report.md` with findings
- [x] Document class imbalance: Happy (7,215) vs Disgust (436) — 16.6x gap
- [x] Commit and push Phase 1 to GitHub

---

## Stage 3 — Full Preprocessing Pipeline `[x] COMPLETE`

> Required by guidelines: resize, normalize, augmentation, train/val/test split, imbalance handling

- [x] **Resize all images to 224x224** for Transfer Learning pipeline (Pipeline B)
      48x48 kept natively for CNN scratch (Pipeline A)
- [x] **Normalize pixel values**: Pipeline A -> [0,1] | Pipeline B -> [-1,1] (MobileNetV2)
- [x] **Data Augmentation** applied to training generators:
  - [x] Random horizontal flip
  - [x] Random rotation (15 deg)
  - [x] Random zoom (10%)
  - [x] Brightness adjustment [0.8, 1.2]
- [x] **Train / Validation / Test Split**:
  - data/train/ -> 85% train (25,888) / 15% val (4,565)
  - data/test/ -> held-out test (7,178)
- [x] **Handle class imbalance**:
  - [x] Disgust augmented x4: 436 -> 2,180 images
  - [x] Class weights computed (disgust=1.9956, happy=0.6030)
  - [x] Saved class_weights.json
- [x] Built `src/full_preprocessor.py` -- dual pipeline module
- [x] Verified batch shapes: A=(32,48,48,1) B=(32,224,224,3)
- [x] Saved class balance comparison plot
- [x] Commit: "Stage 3: Full preprocessing pipeline complete"

---

## Stage 4 — CNN from Scratch `[ ] TODO`

> Build layer by layer — no shortcut model loading

- [ ] Build `src/models/cnn_scratch.py`:
  - [ ] Input layer (48x48x1 or 224x224x1)
  - [ ] Conv2D + ReLU (block 1)
  - [ ] MaxPooling2D
  - [ ] Conv2D + ReLU (block 2)
  - [ ] MaxPooling2D
  - [ ] Conv2D + ReLU (block 3)
  - [ ] MaxPooling2D
  - [ ] Batch Normalization layers
  - [ ] Flatten
  - [ ] Dense (Fully Connected) layers
  - [ ] Dropout (regularization)
  - [ ] Output: Dense(7, activation='softmax')
- [ ] Build `src/trainer.py` — training loop with callbacks:
  - [ ] ModelCheckpoint (save best weights)
  - [ ] EarlyStopping
  - [ ] ReduceLROnPlateau
- [ ] Train CNN scratch model on full preprocessed dataset
- [ ] Save model weights: `models/cnn_scratch_best.h5`
- [ ] Plot and save training curves (loss & accuracy vs epochs)
- [ ] Commit: "Stage 4: CNN from scratch trained"

---

## Stage 5 — Transfer Learning (Built by Hand) `[ ] TODO`

> No drag-and-drop fine-tuning — every layer must be explicitly defined

- [ ] Build `src/models/transfer_learning.py`:
  - [ ] Load pre-trained backbone **without top head** (choose one):
    - VGG16 / ResNet50 / MobileNetV2 — `include_top=False`
  - [ ] **Freeze all base layers** (phase 1: feature extraction only)
  - [ ] Add custom classification head explicitly:
    - [ ] `GlobalAveragePooling2D()` or `Flatten()`
    - [ ] `Dense(256, activation='relu')`
    - [ ] `Dropout(0.5)`
    - [ ] `Dense(128, activation='relu')`
    - [ ] `Dropout(0.3)`
    - [ ] `Dense(7, activation='softmax')`
  - [ ] Train with frozen base (phase 1)
  - [ ] **Unfreeze last N layers** of backbone (phase 2: fine-tuning)
  - [ ] Re-compile with lower learning rate and continue training
- [ ] Save model: `models/transfer_learning_best.h5`
- [ ] Plot training curves for both phases
- [ ] Commit: "Stage 5: Transfer learning model trained"

---

## Stage 6 — Evaluation `[ ] TODO`

> Core graded component — must be thorough with visualizations

- [ ] Build `src/evaluator.py`:
  - [ ] **Accuracy** — overall and per class
  - [ ] **Precision** — macro and per class
  - [ ] **Recall** — macro and per class
  - [ ] **F1-Score** — macro and per class
  - [ ] **Confusion Matrix** — visualized as a labeled heatmap
  - [ ] **Loss & Accuracy Curves** — training vs validation (both models)
  - [ ] **ROC / AUC Curve** (bonus — one-vs-rest for each emotion)
- [ ] Run evaluation for both models on the **held-out test set**
- [ ] Save all evaluation plots to `results/`
- [ ] **Model Comparison Table**:

  | Metric                     | CNN Scratch | Transfer Learning |
  |----------------------------|-------------|-------------------|
  | Train Accuracy             |             |                   |
  | Validation Accuracy        |             |                   |
  | Test Accuracy              |             |                   |
  | Test F1-Score (macro)      |             |                   |
  | Training Time              |             |                   |
  | Trainable Parameters       |             |                   |

- [ ] Commit: "Stage 6: Full evaluation and model comparison"

---

## Stage 7 — Repository Cleanup & Final Structure `[ ] TODO`

- [ ] Reorganize repo to match recommended structure:
  ```
  /data          (gitignored — download via Kaggle)
  /src           (all Python source modules)
  /models        (saved .h5 model weights — gitignored if too large)
  /results       (all plots, confusion matrices, CSV reports)
  /notebooks     (optional — Jupyter exploration notebooks)
  README.md
  requirements.txt
  analytic_report.md
  PROJECT_PLAN.md
  ```
- [ ] Ensure commit history shows incremental progress (not one big commit)
- [ ] Update README with full run instructions
- [ ] Tag release: `v1.0-phase1-complete`, `v2.0-models-complete`, etc.

---

## Stage 8 — Documentation `[ ] TODO`

- [ ] Written report (PDF or Word):
  - [ ] Dataset description and task definition
  - [ ] All preprocessing steps with justification
  - [ ] Architecture of each model (diagram + layer table)
  - [ ] Evaluation results with visualizations
  - [ ] Comparison between CNN scratch and Transfer Learning
  - [ ] Conclusion and recommendations
- [ ] Presentation slides (PowerPoint or PDF):
  - [ ] Dataset overview slide
  - [ ] Preprocessing pipeline slide
  - [ ] Model architecture slides (both models)
  - [ ] Results & comparison slide
  - [ ] Conclusion slide

---

## Stage 9 — Deployment (Bonus +1 to +4) `[ ] OPTIONAL`

- [ ] Build a web application using **Streamlit** (easiest) or **Flask**:
  - [ ] Upload an image (or use webcam)
  - [ ] Run inference using the best saved model
  - [ ] Display predicted emotion + confidence bar chart
  - [ ] Show CLAHE preprocessing effect side-by-side
- [ ] Alternatively: FastAPI REST endpoint
  - `POST /predict` — accepts image, returns `{emotion, confidence}`
- [ ] Document deployment steps in README

---

## Grading Checklist

| Component | Marks | Status |
|-----------|-------|--------|
| Full Project (Code + Dataset + Models + Evaluation) | 10 | `[/]` In progress |
| Presentation & Documentation | 5 | `[ ]` Not started |
| Discussion / Viva | 5 | `[ ]` Upcoming Week 14 |
| Deployment (Bonus) | +1 to +4 | `[ ]` Optional |

---

## Key Warnings & Constraints

> **Prohibited:**
> - Built-in datasets (MNIST, CIFAR, etc.) — We use FER2013 raw files ✓
> - Transfer Learning drag-and-drop (single line fine-tuning) — We build manually ✓
> - Pre-processed datasets — FER2013 raw .jpg files, we process ourselves ✓

> **Critical Issue — Class Imbalance:**
> Disgust (436) vs Happy (7,215) = **16.6x gap** — must be addressed in Stage 3

---

## Decisions — CONFIRMED

| # | Decision | Choice |
|---|----------|--------|
| 1 | Image size | **48x48 grayscale** for CNN scratch · **224x224 RGB** for Transfer Learning |
| 2 | TL backbone | **MobileNetV2** (lightweight, fast, good for 48px upscaled faces) |
| 3 | Imbalance fix | **Class weights + Disgust augmentation multiplier** (combined strategy) |

---

*Last updated: 2026-05-07 | Stage 3 complete, moving to Stage 4 (CNN scratch)*
