# Full Project Walkthrough — Stages 1 to 4
**FER2013 Facial Expression Recognition | CNN from Scratch**

> This document explains every step of the project from raw data download through
> trained model. It is intended for teammates picking up the work at Stage 5.

---

## The Big Picture

We are building a system that looks at a 48×48 pixel grayscale photo of a human
face and decides which of 7 emotions it shows:

```
[Face image] --> [Preprocessing] --> [CNN Model] --> "Happy" / "Sad" / "Angry" ...
```

---

## Stage 1 — Dataset & Project Setup

### What is FER2013?
The **FER2013** dataset is a collection of **28,709 training images** and **7,178
test images**, all:
- 48 × 48 pixels in size
- Grayscale (black and white)
- Each labeled with one of 7 emotions

| Emotion  | Training Count |
|----------|---------------|
| Angry    | 3,995 |
| Disgust  | 436 |
| Fear     | 4,097 |
| Happy    | 7,215 |
| Neutral  | 4,965 |
| Sad      | 4,830 |
| Surprise | 3,171 |
| **Total**| **28,709** |

### How we got it
We used the **Kaggle API** (authenticated with account `ahmed-mahd`) to download
the raw `.jpg` files directly into `data/train/<emotion>/` and `data/test/<emotion>/`.
Nothing was pre-processed — raw images only.

To re-download:
```bash
# Set up credentials: create ~/.kaggle/kaggle.json with your username and key
kaggle datasets download -d msambare/fer2013 -p data/ --unzip
```

### Project structure
```
project/
├── data/
│   ├── train/<emotion>/    <- 28,709 real face images (gitignored)
│   └── test/<emotion>/     <- 7,178 test images (gitignored)
├── src/
│   ├── batch_selector.py
│   ├── quality_checker.py
│   ├── preprocessor.py
│   ├── visualizer.py
│   ├── full_preprocessor.py
│   └── models/
│       └── cnn_scratch.py
├── models/
│   └── cnn_scratch_best.keras  <- saved best model weights
├── output/
│   ├── batch_samples/          <- 210 preprocessed sample images
│   └── reports/                <- all plots, CSVs, JSON summaries
├── main.py                     <- Phase 1 batch analysis runner
├── prepare_data.py             <- Stage 3 pipeline builder
├── train_scratch.py            <- Stage 4 CNN training runner
├── analytics.py                <- Analytic dashboard generator
├── PROJECT_PLAN.md             <- Stage tracker
├── WALKTHROUGH.md              <- This file
└── README.md
```

---

## Stage 2 — Batch Sampling & Exploratory Analysis

### Why sample a batch first?
Before training on all 28,709 images, we validated that the data was usable.
We took a **stratified random sample** of 210 images (30 per emotion) and ran
quality checks on all of them.

### Stratified sampling (`src/batch_selector.py`)
```
data/train/angry/     --> pick 30 random images
data/train/disgust/   --> pick 30 random images
data/train/fear/      --> pick 30 random images
... (same for all 7 classes)
                      --> total: 210 images, equal class representation
```

### Quality checks (`src/quality_checker.py`)
For each image, we checked:

| Check | Threshold |
|-------|-----------|
| Loadable by OpenCV | Must succeed |
| Not blank | Pixel std dev must be > 5 |
| Not too dark | Mean brightness must be > 10 |
| Not too bright | Mean brightness must be < 245 |
| Correct size | Must be 48×48 |
| Grayscale | Must be single channel |

**Result: 210/210 images passed — 100% usable.**

### Preprocessing pipeline (`src/preprocessor.py`)
Each image went through this pipeline:

```
Load as grayscale
      |
Resize to 48x48   (INTER_AREA interpolation — best for downscaling)
      |
CLAHE             (clipLimit=2.0, tileGridSize=8x8)
      |
Gaussian Blur     (3x3 kernel, sigma=0)
      |
Normalize [0,1]   (divide all pixel values by 255)
      |
Output: float32 array, shape (48, 48)
```

**CLAHE** (Contrast Limited Adaptive Histogram Equalization) enhances local
contrast tile-by-tile without blowing out bright regions. This is critical because
FER2013 images have highly variable lighting across subjects and emotions.

**Gaussian Blur** removes high-frequency camera noise and JPEG artifacts before
the model sees the image, preventing the model from learning noise patterns.

**Normalization to [0,1]** keeps gradient computations numerically stable during
backpropagation.

### Key finding
`Happy` has 7,215 images; `Disgust` has only 436 — a **16.6x class imbalance**.
This was flagged here and addressed in Stage 3.

### Outputs
All 6 plots are saved in `output/reports/`:
- `class_distribution.png`
- `before_after_comparison.png`
- `pixel_histograms.png`
- `brightness_distribution.png`
- `sample_grid_raw.png`
- `quality_summary.png`
- `analytics_dashboard.png`

Run:
```bash
python main.py              # Phase 1 batch analysis
python analytics.py         # 9-panel analytic dashboard
```

---

## Stage 3 — Full Preprocessing Pipeline

Stage 2 was exploratory analysis on 210 images. Stage 3 built the real
production pipelines for training on all 28,709 images.

### Two pipelines (`src/full_preprocessor.py`)

| | Pipeline A | Pipeline B |
|-|------------|------------|
| **For** | CNN from Scratch | Transfer Learning (MobileNetV2) |
| **Image size** | 48×48 | 224×224 |
| **Color** | Grayscale | RGB |
| **Normalization** | [0, 1] divide by 255 | [-1, 1] via `mobilenet_v2.preprocess_input` |
| **Input tensor shape** | (batch, 48, 48, 1) | (batch, 224, 224, 3) |

### Fixing class imbalance — two strategies combined

**Strategy 1: Disgust augmentation (disk-level)**
We generated 4 augmented copies of every Disgust image before training:
```
436 original Disgust images
  x 4 augmented copies (flip, rotation, zoom, brightness shift)
= 1,744 new images written to disk
= 2,180 total Disgust images
```
Function: `augment_minority_class()` in `src/full_preprocessor.py`
To re-run: `python prepare_data.py`
To clean: `python prepare_data.py --clean`

**Strategy 2: Class weights (training-level)**
Inverse-frequency weights force the model to penalise minority class errors more:

| Class | Weight | Stored in |
|-------|--------|-----------|
| Angry | 1.089 | `output/reports/class_weights.json` |
| Disgust | 1.996 | |
| Fear | 1.062 | |
| Happy | 0.603 | |
| Neutral | 0.876 | |
| Sad | 0.901 | |
| Surprise | 1.372 | |

Passed to `model.fit(class_weight=class_weights)` at training time.

### Data augmentation (training batches only)
Applied live by `ImageDataGenerator` — the model never sees the same augmented
version twice:

```
Random horizontal flip  (50% chance)
Random rotation         (up to ±15 degrees)
Random zoom             (up to 10%)
Random horizontal/vertical shift  (up to 10%)
Random brightness       (80% to 120% of original)
```

Validation and test sets receive **no augmentation** — only normalization.

### Train / Validation / Test split
```
data/train/  (30,453 images after Disgust augmentation)
    85% --> training set    : 25,888 images
    15% --> validation set  :  4,565 images

data/test/   (7,178 images — never seen during training)
    --> held-out test set
```

### Verified batch shapes
```
Pipeline A: X=(32,48,48,1)   float32   range [0.000, 1.000]  ✓
Pipeline B: X=(32,224,224,3) float32   range [-1.000, 1.000] ✓
All 7 classes represented in first batch: ✓
```

Run:
```bash
python prepare_data.py          # builds both pipelines, runs augmentation
python prepare_data.py --clean  # removes augmented images (to regenerate fresh)
```

---

## Stage 4 — CNN from Scratch

### Architecture (`src/models/cnn_scratch.py`)

Input: **(48, 48, 1)** — one grayscale face image

```
BLOCK 1                              Output shape
  Conv2D(32, 3x3, padding=same)      (48, 48, 32)
  BatchNormalization
  ReLU
  Conv2D(32, 3x3, padding=same)      (48, 48, 32)
  BatchNormalization
  ReLU
  MaxPooling2D(2x2)                  (24, 24, 32)
  Dropout(0.25)

BLOCK 2
  Conv2D(64, 3x3, padding=same)      (24, 24, 64)
  BatchNormalization + ReLU
  Conv2D(64, 3x3, padding=same)      (24, 24, 64)
  BatchNormalization + ReLU
  MaxPooling2D(2x2)                  (12, 12, 64)
  Dropout(0.25)

BLOCK 3
  Conv2D(128, 3x3, padding=same)     (12, 12, 128)
  BatchNormalization + ReLU
  Conv2D(128, 3x3, padding=same)     (12, 12, 128)
  BatchNormalization + ReLU
  MaxPooling2D(2x2)                  ( 6,  6, 128)
  Dropout(0.25)

BLOCK 4
  Conv2D(256, 3x3, padding=same)     ( 6,  6, 256)
  BatchNormalization + ReLU
  Conv2D(256, 3x3, padding=same)     ( 6,  6, 256)
  BatchNormalization + ReLU
  MaxPooling2D(2x2)                  ( 3,  3, 256)
  Dropout(0.25)

FLATTEN                              (2304,)

FC HEAD
  Dense(512)                         (512,)
  BatchNormalization + ReLU
  Dropout(0.50)

  Dense(256, activation=relu)        (256,)
  Dropout(0.30)

OUTPUT
  Dense(7, activation=softmax)       (7,)
  --> [0.02, 0.01, 0.04, 0.74, 0.08, 0.06, 0.05]
       angry  disg  fear  happy neut  sad   surp
```

**Total trainable parameters: 2,486,439 (~9.5 MB)**

### Training setup (`train_scratch.py`)

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss function | Categorical Cross-Entropy |
| Batch size | 32 |
| Max epochs | 60 |
| EarlyStopping patience | 12 epochs |
| ReduceLROnPlateau patience | 6 epochs, factor 0.5 |
| Class weights | Yes (from `class_weights.json`) |

### Training callbacks

| Callback | Monitors | Action |
|----------|----------|--------|
| `ModelCheckpoint` | val_accuracy | Save best weights to `models/cnn_scratch_best.keras` |
| `EarlyStopping` | val_accuracy | Stop if no improvement for 12 epochs; restore best weights |
| `ReduceLROnPlateau` | val_loss | Halve LR if no improvement for 6 epochs |
| `CSVLogger` | — | Write all metrics to `output/reports/scratch_training_log.csv` |

### Training results

| Epoch | Train Acc | Val Acc | Event |
|-------|-----------|---------|-------|
| 1 | 18.5% | 17.7% | Start (~random) |
| 10 | 55.3% | 50.9% | Learning faces |
| 22 | 61.1% | 59.3% | Strong progress |
| 47 | 67.0% | 63.5% | **LR reduced 0.001→0.0005** |
| **57** | **69.1%** | **65.8%** | **Best val accuracy — model saved** |
| 60 | 69.4% | 65.4% | EarlyStopping fires, best weights restored |

### Final evaluation on held-out test set

```
Test Accuracy : 66.27%
Test Loss     : 0.9694
Parameters    : 2,489,383
Training time : ~102 minutes (CPU only)
```

Saved files:
```
models/cnn_scratch_best.keras             <- best model weights
output/reports/scratch_training_log.csv   <- per-epoch metrics
output/reports/scratch_training_curves.png <- loss & accuracy plots
output/reports/scratch_summary.json       <- final summary stats
```

Run:
```bash
python train_scratch.py              # train from scratch (default 60 epochs)
python train_scratch.py --epochs 30  # shorter run
python train_scratch.py --no-weights # disable class weighting
```

### How to load and use the trained model

```python
import tensorflow as tf
import numpy as np
import cv2

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load saved model
model = tf.keras.models.load_model("models/cnn_scratch_best.keras")

# Load and preprocess a face image
img = cv2.imread("path/to/face.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48, 48))

# Apply CLAHE (same as training pipeline)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = clahe.apply(img)

# Normalize and reshape
img = img.astype("float32") / 255.0
img = img.reshape(1, 48, 48, 1)     # add batch and channel dims

# Predict
probs = model.predict(img)[0]       # shape: (7,)
predicted_emotion = EMOTIONS[np.argmax(probs)]
confidence = probs.max()

print(f"Prediction: {predicted_emotion}  ({confidence*100:.1f}% confidence)")
```

---

## What's Next — Stages 5 to 9

| Stage | Task | Script to create |
|-------|------|-----------------|
| **5** | Transfer Learning (MobileNetV2) | **Done** |
| **6** | Evaluation: confusion matrix, F1, ROC | **Done** (`src/evaluator.py`) |
| **7** | Repo cleanup + final structure | — |
| **8** | Written report (PDF) + slides | — |
| **9** | Deployment: Streamlit web app (bonus) | `app.py` |

---

## Stage 5 & 6 — Transfer Learning & Evaluation

### Stage 5: Transfer Learning (MobileNetV2)
- Added `src/models/transfer_learning.py` using Keras `MobileNetV2` with `include_top=False`.
- The base model is frozen initially, and a custom classification head (`GlobalAveragePooling2D` -> `Dense(256)` -> `Dense(128)` -> `Dense(7)`) is added.
- Added `train_transfer.py` which trains the model in two phases:
  - **Phase 1**: Trains only the custom top layers using `Adam(1e-3)`.
  - **Phase 2**: Unfreezes the top 20 layers of the base model and fine-tunes with a lower learning rate `Adam(1e-5)`.

### Stage 6: Evaluation
- Added `src/evaluator.py`.
- Generates accuracy, precision, recall, and F1-score for both the `CNN_Scratch` and `MobileNetV2` models.
- Plots confusion matrices to visually compare the misclassifications.
- Note: Requires the full FER2013 dataset downloaded in `data/train/` to execute successfully.

---

## Key Design Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Image size (scratch) | 48×48 | Native FER2013 size, no upscaling needed |
| Image size (TL) | 224×224 | MobileNetV2 was trained at this resolution |
| TL backbone | MobileNetV2 | Lightweight, fast, designed for mobile/embedded — good fit for small 48px faces upscaled |
| Imbalance fix | Class weights + Disgust augmentation | Combined approach handles both training-time and data-level imbalance |
| CLAHE over plain histogram eq | CLAHE | Local adaptive enhancement avoids blowing out already-bright regions |
| Dropout rates | 0.25 conv / 0.50 fc1 / 0.30 fc2 | Higher dropout in the fully-connected head where overfitting risk is greatest |
| EarlyStopping patience | 12 | Long enough to survive the oscillations caused by augmentation randomness |

---

*Document generated: 2026-05-09 | Stages 1-6 complete/implemented*
