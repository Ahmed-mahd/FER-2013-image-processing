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
  - [x] Fixed: pandas/tqdm moved to top-level imports (linter `pd` warning resolved)
- [x] Run quality checks on batch → 210/210 usable (100%)
- [x] Save quality report CSV (`output/reports/quality_report.csv`)
- [x] Build `src/visualizer.py` — 6 analysis plots:
  - [x] Class distribution bar chart
  - [x] Brightness violin plot per class
  - [x] Raw sample grid (before preprocessing)
  - [x] Before vs After comparison (CLAHE effect)
  - [x] Pixel intensity histograms (post-preprocessing)
  - [x] Usability quality summary chart
  - [x] Fixed: axes crash when n_rows==1 (np.atleast_2d guard added)
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
- [x] Built `src/full_preprocessor.py` — dual pipeline module
  - [x] Fixed: `build_tl_pipeline` now returns `(train_ds, val_ds, test_ds, train_steps, val_steps)`
  - [x] Fixed: uses `math.ceil` for correct step count
- [x] Built `src/preprocessor.py` — single-image pipeline (CLAHE + blur + normalize)
  - [x] Fixed: pandas/tqdm moved to top-level imports
- [x] Verified batch shapes: A=(32,48,48,1) B=(32,224,224,3)
- [x] Saved class balance comparison plot
- [x] Commit: "Stage 3: Full preprocessing pipeline complete"

---

## Stage 4 — CNN from Scratch `[x] COMPLETE`

> Build layer by layer — no shortcut model loading

- [x] Build `src/models/cnn_scratch.py`:
  - [x] Input layer (48x48x1)
  - [x] Conv2D + BN + ReLU block 1 (32 filters)
  - [x] MaxPooling2D + Dropout(0.25)
  - [x] Conv2D + BN + ReLU block 2 (64 filters)
  - [x] MaxPooling2D + Dropout(0.25)
  - [x] Conv2D + BN + ReLU block 3 (128 filters)
  - [x] MaxPooling2D + Dropout(0.25)
  - [x] Conv2D + BN + ReLU block 4 (256 filters)
  - [x] MaxPooling2D + Dropout(0.25)
  - [x] Flatten (2,304 units)
  - [x] Dense(512) + BN + ReLU + Dropout(0.50)
  - [x] Dense(256) + ReLU + Dropout(0.30)
  - [x] Output Dense(7, softmax)
- [x] Build `train_scratch.py` with callbacks:
  - [x] ModelCheckpoint (save best val_accuracy)
  - [x] EarlyStopping (patience=12)
  - [x] ReduceLROnPlateau (patience=6, factor=0.5)
  - [x] CSVLogger
- [x] Train model — 60 epochs, LR reduced at epoch 47 (0.001->0.0005)
- [x] Save model: `models/cnn_scratch_best.keras`
- [x] Plot training curves: `output/reports/scratch_training_curves.png`
- [x] Save summary: `output/reports/scratch_summary.json`

**Results:**
- Total parameters    : 2,489,383
- Best val accuracy   : 65.83% (epoch 57)
- Test accuracy       : **66.27%** ← baseline to beat
- Training time       : ~102 min (CPU)

---

## Stage 5 — Transfer Learning (EfficientNetB0) `[x] COMPLETE (needs retrain)`

> No drag-and-drop fine-tuning — every layer explicitly defined

- [x] Build `src/models/transfer_learning.py`:
  - [x] EfficientNetB0 backbone loaded without top (`include_top=False`, `weights='imagenet'`)
  - [x] Freeze all base layers (Phase 1: feature extraction only)
  - [x] Custom classification head explicitly defined:
    - [x] `GlobalAveragePooling2D()`
    - [x] `Dense(256, relu)` + BatchNormalization + `Dropout(0.5)`
    - [x] `Dense(128, relu)` + `Dropout(0.3)`
    - [x] `Dense(7, softmax)`
  - [x] Phase 2: `unfreeze_top_layers()` unfreezes last 50 layers (BN kept frozen)
- [x] Build `train_transfer.py` with two-phase training:
  - [x] Phase 1: LR=1e-3, epochs=25 (head only)
  - [x] Phase 2: LR=1e-5, epochs=30 (fine-tuning last 50 layers)
  - [x] All callbacks: ModelCheckpoint, EarlyStopping(patience=8), ReduceLROnPlateau, CSVLogger
  - [x] Class weights applied
- [x] Save model: `models/transfer_learning_best.keras`
- [x] Plot training curves: `output/reports/transfer_training_curves.png`
- [x] Save summary: `output/reports/transfer_summary.json`

**Why EfficientNetB0 over MobileNetV2:**
- MobileNetV2 tested but peaked at 64% (couldn't break CNN scratch baseline of 66.3%)
- EfficientNetB0: compound scaling (width+depth+resolution) → better fine-grained feature maps
- Expected: **70–74% test accuracy**
- Same API, zero extra dependencies

**⚠ ACTION NEEDED:** Teammate must `git pull` and run:
```bash
python train_transfer.py    # defaults: --epochs1 25 --epochs2 30 --unfreeze 50
```

---

## Stage 6 — Evaluation `[/] IN PROGRESS`

> Core graded component — must be thorough with visualizations

- [x] Build `src/evaluator.py` (fully rewritten):
  - [x] **Accuracy** — overall and per class
  - [x] **Precision, Recall, F1-Score** — macro and per class (`classification_report`)
  - [x] **Confusion Matrix** — dark-themed heatmap saved as PNG
  - [x] `predict_from_generator()` for CNN Scratch (Pipeline A — ImageDataGenerator)
  - [x] `predict_from_tf_dataset()` for MobileNetV2 (Pipeline B — tf.data)
  - [x] **Model Comparison Table** printed to terminal
  - [x] Saves `output/reports/evaluation_results.json`
- [ ] **Run evaluation** — both models on held-out test set (7,178 images)
  - Blocked by: Transfer Learning retrain must complete first
- [ ] ROC / AUC Curve (one-vs-rest per emotion) — bonus visualization

**To run:**
```bash
python train_transfer.py          # retrain TL model with fixed bugs first
python src/evaluator.py           # then generate full comparison
```

**Model Comparison Table (fill after retrain):**

| Metric | CNN Scratch | Transfer Learning |
|--------|-------------|-------------------|
| Test Accuracy | **66.27%** | TBD (expected 68–73%) |
| Best Val Accuracy | 65.83% | TBD |
| Test F1-Score (macro) | TBD | TBD |
| Training Time | ~102 min | TBD |
| Trainable Parameters | 2,489,383 | ~3.4M (Phase 2) |

---

## Stage 7 — Repository Cleanup & Final Structure `[ ] TODO`

- [ ] Verify commit history shows incremental progress (not one big commit)
  - Current: 7 clean commits already ✓ — no action needed
- [ ] Update README with full run instructions (step-by-step from clone to results)
- [ ] Add git tags: `v1.0-data-pipeline`, `v2.0-cnn-scratch`, `v3.0-transfer-learning`
- [ ] Move all result images to consistent `output/reports/` directory (already done ✓)

---

## Stage 8 — Documentation `[ ] TODO`

- [ ] Written report (PDF):
  - [ ] Dataset description and task definition
  - [ ] Preprocessing steps with justification (CLAHE, augmentation, class weights)
  - [ ] Architecture of each model with layer table
  - [ ] Evaluation results with confusion matrix screenshots
  - [ ] Comparison between CNN scratch and Transfer Learning
  - [ ] Conclusion and recommendations
- [ ] Presentation slides:
  - [ ] Dataset overview slide
  - [ ] Preprocessing pipeline slide
  - [ ] CNN Scratch architecture slide
  - [ ] Transfer Learning (MobileNetV2) architecture slide
  - [ ] Results & comparison slide
  - [ ] Conclusion slide

---

## Stage 9 — Deployment (Bonus +1 to +4) `[x] COMPLETE`

- [x] Built `app.py` — Streamlit web application:
  - [x] Upload any face image (jpg, png, webp)
  - [x] Model selector: CNN Scratch or EfficientNetB0 Transfer Learning
  - [x] Applies exact same preprocessing pipeline used during training
  - [x] Shows predicted emotion with emoji + confidence percentage
  - [x] Full confidence bar chart for all 7 emotions
  - [x] Before/After CLAHE visualization for CNN Scratch model
  - [x] Dark-themed premium UI

**Run:**
```bash
pip install streamlit
streamlit run app.py
```

---

## Grading Checklist

| Component | Marks | Status |
|-----------|-------|--------|
| Full Project (Code + Dataset + Models + Evaluation) | 10 | `[/]` Stage 5 retrain in progress |
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
> Disgust (436) vs Happy (7,215) = **16.6x gap** — addressed via class weights + disk augmentation ✓

---

## Decisions — CONFIRMED

| # | Decision | Choice |
|---|----------|--------|
| 1 | Image size | **48x48 grayscale** for CNN scratch · **224x224 RGB** for Transfer Learning |
| 2 | TL backbone | **EfficientNetB0** (switched from MobileNetV2 which peaked at 64%) |
| 3 | Imbalance fix | **Class weights + Disgust augmentation x4** (combined strategy) |
| 4 | Evaluation pipeline | Separate predict functions for Generator vs tf.data |
| 5 | steps_per_epoch | Returned directly from `build_tl_pipeline()` (no manual counting) |
| 6 | Deployment | Streamlit app `app.py` — supports both models |

---

*Last updated: 2026-05-12 | Stages 1–5 code complete. Stage 5 needs retrain. Stage 6 ready to run after retrain.*
