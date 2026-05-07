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

## Stage 3 — Full Preprocessing Pipeline `[ ] TODO`

> Required by guidelines: resize, normalize, augmentation, train/val/test split, imbalance handling

- [ ] **Resize all images to 224x224** (model-ready size, not just 48x48 batch sample)
  - Note: FER2013 is natively 48x48 — decide: keep 48x48 or upscale to 224x224 for transfer learning
- [ ] **Normalize pixel values** to `[0, 1]` (already done for batch; apply to full dataset)
- [ ] **Data Augmentation** (use `ImageDataGenerator` or `albumentations`):
  - [ ] Random horizontal flip
  - [ ] Random rotation (±15°)
  - [ ] Random zoom (10%)
  - [ ] Brightness adjustment
- [ ] **Train / Validation / Test Split** (70% / 15% / 15%)
  - `data/train/` → 70% train, 15% val (split from train folder)
  - Use `data/test/` as held-out test set
- [ ] **Handle class imbalance** (choose one or combine):
  - [ ] Option A: Class weights (inverse frequency weighting in loss function)
  - [ ] Option B: Oversample minority class (Disgust) via augmentation
  - [ ] Document chosen strategy and justify it
- [ ] Build `src/full_preprocessor.py` — applies pipeline to entire dataset
- [ ] Save preprocessed dataset stats and verify with plots
- [ ] Commit: "Stage 3: Full preprocessing pipeline complete"

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

## Decisions to Make Before Stage 3

1. **Image size for models:** Keep 48x48 (native FER2013) or resize to 224x224 for transfer learning?
   - Recommendation: Use 48x48 for CNN scratch, 224x224 for transfer learning
2. **Transfer Learning backbone:** VGG16 vs ResNet50 vs MobileNetV2?
   - Recommendation: MobileNetV2 (lighter, faster, good for small images)
3. **Imbalance strategy:** Class weights vs augmentation-based oversampling?
   - Recommendation: Combine both — class weights + augment Disgust class

---

*Last updated: 2026-05-07 | Phase 1 complete, moving to Stage 3*
