# FER2013 — Facial Expression Recognition

A complete deep learning pipeline that classifies 48×48 grayscale face images into **7 emotions** using the FER2013 dataset.

**Models:** CNN from Scratch (66.2%) vs EfficientNetB0 Transfer Learning (63.2%)

---

## Results

| Metric | CNN Scratch | EfficientNetB0 |
|--------|:-----------:|:--------------:|
| Test Accuracy | **66.23%** | 63.22% |
| Macro Precision | 68.05% | 62.68% |
| Macro Recall | 58.87% | 60.41% |
| Macro F1-Score | 59.77% | 60.80% |
| Training Time | ~102 min (CPU) | ~39 min (GPU) |
| Trainable Params | 2,489,383 | 362,247 (Ph1) / 4.4M (Ph2) |

---

## Project Structure

```
FER-2013-image-processing/
├── data/
│   ├── train/<emotion>/          # 28,709 training images
│   └── test/<emotion>/           # 7,178 test images
├── models/
│   ├── cnn_scratch_best.keras    # Stage 4 — CNN Scratch
│   └── transfer_learning_best.keras  # Stage 5 — EfficientNetB0
├── output/reports/
│   ├── stage2_eda/               # Quality checks, histograms, class distribution
│   ├── stage3_preprocessing/     # CLAHE comparison, augmentation samples, class weights
│   ├── stage4_cnn_scratch/       # Training curves, log, summary JSON
│   ├── stage5_transfer_learning/ # Training curves, log, summary JSON
│   └── stage6_evaluation/        # Confusion matrices, ROC curves, comparison chart
├── src/
│   ├── models/
│   │   ├── cnn_scratch.py        # CNN architecture (from scratch)
│   │   └── transfer_learning.py  # EfficientNetB0 + custom head
│   ├── full_preprocessor.py      # Pipeline A (48×48 grayscale) + Pipeline B (224×224 RGB)
│   ├── evaluator.py              # Stage 6 — evaluation & comparison
│   ├── visualizer.py             # Plotting utilities
│   └── quality_checker.py        # Stage 2 — dataset quality checks
├── app.py                        # Streamlit demo app (Stage 9)
├── train_scratch.py              # Stage 4 training script
├── train_transfer.py             # Stage 5 training script
├── prepare_data.py               # Stage 3 — augmentation + class weights
├── analytics.py                  # Stage 2 — EDA
└── requirements.txt
```

---

## Quick Start (from scratch)

### 1. Clone & install

```bash
git clone https://github.com/Ahmed-mahd/FER-2013-image-processing.git
cd FER-2013-image-processing
pip install -r requirements.txt
```

### 2. Download FER2013 dataset

```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/
# Ensure structure: data/train/<emotion>/ and data/test/<emotion>/
```

### 3. Prepare data (augmentation + class weights)

```bash
python prepare_data.py
```

### 4. Train CNN from Scratch

```bash
python train_scratch.py
# Outputs → output/reports/stage4_cnn_scratch/
# Model   → models/cnn_scratch_best.keras
```

### 5. Train Transfer Learning (EfficientNetB0)

```bash
# With GPU (recommended):
NVIDIA_BASE="$HOME/.local/lib/python3.12/site-packages/nvidia"
CUDA_LIBS=$(find "$NVIDIA_BASE" -type d -name "lib" | tr '\n' ':')
LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}" python train_transfer.py

# Without GPU:
python train_transfer.py
# Outputs → output/reports/stage5_transfer_learning/
# Model   → models/transfer_learning_best.keras
```

### 6. Evaluate both models

```bash
python src/evaluator.py
# Outputs → output/reports/stage6_evaluation/
#   - cnn_scratch_confusion_matrix.png
#   - cnn_scratch_roc_curves.png
#   - efficientnetb0_confusion_matrix.png
#   - efficientnetb0_roc_curves.png
#   - model_comparison.png
#   - evaluation_results.json
```

### 7. Run the Streamlit demo app

```bash
streamlit run app.py
# Open: http://localhost:8501
# Upload any face photo → get emotion prediction
```

---

## GPU Setup (NVIDIA RTX 3050 Ti / any NVIDIA GPU)

If TensorFlow doesn't detect your GPU:

```bash
# Add to ~/.bashrc for persistent fix:
NVIDIA_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia
CUDA_LIBS=$(find "$NVIDIA_SITE" -type d -name "lib" 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH}"
```

---

## Environment

- Python 3.12
- TensorFlow 2.x (with CUDA 12.x / cuDNN 9.x)
- See `requirements.txt` for full dependency list
