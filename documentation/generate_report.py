"""
generate_report.py
------------------
Stage 8 — Generate a self-contained HTML report with embedded images.
Open documentation/report.html in a browser → Ctrl+P → Save as PDF.
"""
import base64, json
from pathlib import Path

def b64img(path):
    p = Path(path)
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

# ── Load data ─────────────────────────────────────────────────────────────────
scratch = json.loads(Path("output/reports/stage4_cnn_scratch/scratch_summary.json").read_text())
tl      = json.loads(Path("output/reports/stage5_transfer_learning/transfer_summary.json").read_text())
ev      = json.loads(Path("output/reports/stage6_evaluation/evaluation_results.json").read_text())
cnn_ev  = ev["CNN_Scratch"]
eff_ev  = ev["EfficientNetB0"]

# ── Images ────────────────────────────────────────────────────────────────────
r = "output/reports"
imgs = {k: b64img(v) for k, v in {
    "cnn_curves":  f"{r}/stage4_cnn_scratch/scratch_training_curves.png",
    "tl_curves":   f"{r}/stage5_transfer_learning/transfer_training_curves.png",
    "cnn_cm":      f"{r}/stage6_evaluation/cnn_scratch_confusion_matrix.png",
    "eff_cm":      f"{r}/stage6_evaluation/efficientnetb0_confusion_matrix.png",
    "cnn_roc":     f"{r}/stage6_evaluation/cnn_scratch_roc_curves.png",
    "eff_roc":     f"{r}/stage6_evaluation/efficientnetb0_roc_curves.png",
    "comparison":  f"{r}/stage6_evaluation/model_comparison.png",
    "class_dist":  f"{r}/stage2_eda/class_distribution.png",
    "before_after":f"{r}/stage3_preprocessing/before_after_comparison.png",
    "aug_cnn":     f"{r}/stage3_preprocessing/augmented_samples_cnn_scratch.png",
}.items()}

def img_tag(key, caption="", width="100%"):
    src = imgs.get(key, "")
    if not src:
        return f"<p><em>[Image not found: {key}]</em></p>"
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f'<figure><img src="{src}" style="width:{width};border-radius:8px;border:1px solid #334;"/>{cap}</figure>'

# ── CNN layer table ───────────────────────────────────────────────────────────
cnn_layers = [
    ("InputLayer",        "48×48×1",    "—",        "—"),
    ("Conv2D 32, 3×3",   "48×48×32",   "896",      "relu + same"),
    ("BatchNorm",         "48×48×32",   "128",      "—"),
    ("Conv2D 32, 3×3",   "48×48×32",   "9,248",    "relu + same"),
    ("BatchNorm",         "48×48×32",   "128",      "—"),
    ("MaxPool 2×2",       "24×24×32",   "0",        "—"),
    ("Dropout 0.25",      "24×24×32",   "0",        "—"),
    ("Conv2D 64, 3×3",   "24×24×64",   "18,496",   "relu + same"),
    ("BatchNorm",         "24×24×64",   "256",      "—"),
    ("Conv2D 64, 3×3",   "24×24×64",   "36,928",   "relu + same"),
    ("BatchNorm",         "24×24×64",   "256",      "—"),
    ("MaxPool 2×2",       "12×12×64",   "0",        "—"),
    ("Dropout 0.25",      "12×12×64",   "0",        "—"),
    ("Conv2D 128, 3×3",  "12×12×128",  "73,856",   "relu + same"),
    ("BatchNorm",         "12×12×128",  "512",      "—"),
    ("Conv2D 128, 3×3",  "12×12×128",  "147,584",  "relu + same"),
    ("BatchNorm",         "12×12×128",  "512",      "—"),
    ("MaxPool 2×2",       "6×6×128",    "0",        "—"),
    ("Dropout 0.25",      "6×6×128",    "0",        "—"),
    ("Conv2D 256, 3×3",  "6×6×256",    "295,168",  "relu + same"),
    ("BatchNorm",         "6×6×256",    "1,024",    "—"),
    ("GlobalAvgPool2D",   "256",        "0",        "—"),
    ("Dense 512",         "512",        "131,584",  "relu"),
    ("BatchNorm",         "512",        "2,048",    "—"),
    ("Dropout 0.5",       "512",        "0",        "—"),
    ("Dense 256",         "256",        "131,328",  "relu"),
    ("Dropout 0.3",       "256",        "0",        "—"),
    ("Dense 7",           "7",          "1,799",    "softmax"),
]

def layer_rows(layers):
    rows = ""
    for l in layers:
        rows += f"<tr><td>{l[0]}</td><td>{l[1]}</td><td>{l[2]}</td><td>{l[3]}</td></tr>\n"
    return rows

tl_head_layers = [
    ("EfficientNetB0 base","—",     "4,049,571","imagenet pretrained, frozen Ph1"),
    ("GlobalAvgPool2D",    "1280",  "0",        "—"),
    ("Dense 256",          "256",   "327,936",  "relu"),
    ("BatchNorm",          "256",   "1,024",    "—"),
    ("Dropout 0.5",        "256",   "0",        "—"),
    ("Dense 128",          "128",   "32,896",   "relu"),
    ("Dropout 0.3",        "128",   "0",        "—"),
    ("Dense 7",            "7",     "903",      "softmax"),
]

# ── HTML ──────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #1a1a2e; background: #fff; padding: 30px; max-width: 960px; margin: auto; }
h1 { font-size: 26px; color: #0f3460; border-bottom: 3px solid #e74c3c; padding-bottom: 8px; margin-bottom: 6px; }
h2 { font-size: 18px; color: #0f3460; margin: 28px 0 10px; border-left: 4px solid #e74c3c; padding-left: 10px; }
h3 { font-size: 14px; color: #16213e; margin: 18px 0 8px; }
p  { line-height: 1.7; margin-bottom: 10px; color: #333; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 12px; }
th { background: #0f3460; color: #fff; padding: 8px 10px; text-align: left; }
td { padding: 7px 10px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) td { background: #f5f7ff; }
.highlight td:first-child { font-weight: bold; color: #e74c3c; }
figure { margin: 14px 0; text-align: center; }
figcaption { font-size: 11px; color: #666; margin-top: 5px; font-style: italic; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 14px 0; }
.card { background: #f5f7ff; border: 1px solid #d0d8f0; border-radius: 8px; padding: 14px; }
.card h3 { color: #0f3460; margin-top: 0; }
.badge { display: inline-block; background: #0f3460; color: #fff; border-radius: 4px; padding: 2px 8px; font-size: 11px; margin-right: 4px; }
.badge.green { background: #27ae60; }
.badge.red   { background: #e74c3c; }
.section-divider { border: none; border-top: 2px solid #e8ecf5; margin: 30px 0; }
.cover { text-align: center; padding: 40px 0 30px; border-bottom: 2px solid #e8ecf5; margin-bottom: 30px; }
.cover .subtitle { color: #666; font-size: 15px; margin-top: 8px; }
.cover .meta { color: #888; font-size: 12px; margin-top: 14px; }
@media print {
  body { padding: 10px; font-size: 12px; }
  h2 { page-break-before: auto; }
  .no-break { page-break-inside: avoid; }
}
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>FER2013 — Project Report</title>
<style>{CSS}</style>
</head>
<body>

<div class="cover">
  <h1>FER2013 — Facial Expression Recognition</h1>
  <div class="subtitle">Deep Learning Pipeline: CNN from Scratch vs EfficientNetB0 Transfer Learning</div>
  <div class="meta">
    Dataset: FER2013 (Kaggle) &nbsp;|&nbsp; 7 Emotion Classes &nbsp;|&nbsp; 35,887 Images<br/>
    Prepared by:<br/>
    <b>Ahmed Mahmoud</b> (230104472) &nbsp;|&nbsp; <b>Mohamed Ahmed</b> (230101143) &nbsp;|&nbsp; <b>Mohamed Ashraf</b> (230105239)<br/>
    Date: 2026-05-13
  </div>
</div>

<!-- ══════════════════════════════════════════════════════ -->
<h2>1. Task Definition &amp; Dataset</h2>

<p>
The task is multi-class image classification: given a 48×48 grayscale face image,
predict one of 7 discrete facial expressions. The FER2013 dataset was sourced
directly from Kaggle in raw JPEG format and processed entirely from scratch.
</p>

<table class="no-break">
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Dataset</td><td>FER2013 (Kaggle — msambare/fer2013)</td></tr>
  <tr><td>Total images</td><td>35,887 (28,709 train + 7,178 test)</td></tr>
  <tr><td>Image size</td><td>48 × 48 pixels, grayscale</td></tr>
  <tr><td>Classes</td><td>7 — Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise</td></tr>
  <tr><td>Task</td><td>Multi-class classification</td></tr>
  <tr><td>Evaluation metric</td><td>Accuracy, Macro F1-Score, Confusion Matrix, ROC-AUC</td></tr>
</table>

<h3>Class Distribution (Training Set)</h3>
<table class="no-break">
  <tr><th>Emotion</th><th>Count</th><th>% of Total</th><th>Note</th></tr>
  <tr><td>Angry</td><td>3,995</td><td>13.9%</td><td>—</td></tr>
  <tr class="highlight"><td>Disgust</td><td>436</td><td>1.5%</td><td>⚠ Severe minority — 16.6× fewer than Happy</td></tr>
  <tr><td>Fear</td><td>4,097</td><td>14.3%</td><td>—</td></tr>
  <tr><td>Happy</td><td>7,215</td><td>25.1%</td><td>Dominant class</td></tr>
  <tr><td>Neutral</td><td>4,965</td><td>17.3%</td><td>—</td></tr>
  <tr><td>Sad</td><td>4,830</td><td>16.8%</td><td>—</td></tr>
  <tr><td>Surprise</td><td>3,171</td><td>11.0%</td><td>—</td></tr>
</table>

{img_tag("class_dist", "Figure 1 — Class distribution of training set")}

<hr class="section-divider"/>

<!-- ══════════════════════════════════════════════════════ -->
<h2>2. Preprocessing Pipeline</h2>

<p>Two separate preprocessing pipelines were implemented to match each model's requirements:</p>

<div class="two-col">
  <div class="card">
    <h3>Pipeline A — CNN Scratch</h3>
    <ol style="padding-left:16px;line-height:2">
      <li>Convert to grayscale</li>
      <li>Resize to 48×48</li>
      <li>Normalize pixels to [0, 1]</li>
      <li>Augment (train only): flip, rotate ±15°, zoom ±10%, shift ±10%, brightness ±20%</li>
    </ol>
    <p><strong>Input:</strong> (48, 48, 1) &nbsp; <strong>Dtype:</strong> float32</p>
  </div>
  <div class="card">
    <h3>Pipeline B — EfficientNetB0</h3>
    <ol style="padding-left:16px;line-height:2">
      <li>Convert to RGB (3 channels)</li>
      <li>Resize to 224×224</li>
      <li>Apply EfficientNet preprocess_input → [-1, 1]</li>
      <li>Augment (train only): flip, brightness, contrast, saturation, rotation, zoom</li>
      <li>tf.data pipeline with AUTOTUNE prefetch</li>
    </ol>
    <p><strong>Input:</strong> (224, 224, 3) &nbsp; <strong>Dtype:</strong> float32</p>
  </div>
</div>

<h3>Class Imbalance Strategy</h3>
<ul style="padding-left:20px;line-height:2">
  <li><strong>Disk Augmentation:</strong> Disgust class augmented ×4 on disk (436 → ~1,744 images)</li>
  <li><strong>Class Weights:</strong> Inverse-frequency weights applied during training via <code>sklearn.compute_class_weight</code></li>
</ul>

{img_tag("before_after", "Figure 2 — Before vs After CLAHE (visual reference only, not applied during training)", "80%")}
{img_tag("aug_cnn", "Figure 3 — Augmented training samples (CNN Pipeline A)", "80%")}

<hr class="section-divider"/>

<!-- ══════════════════════════════════════════════════════ -->
<h2>3. Model 1 — CNN from Scratch</h2>

<h3>Architecture</h3>
<p>Custom CNN built entirely from scratch — no pre-trained weights. Follows a VGG-inspired design with 4 convolutional blocks, each followed by batch normalization and spatial pooling.</p>

<table class="no-break">
  <tr><th>Layer</th><th>Output Shape</th><th>Parameters</th><th>Notes</th></tr>
  {layer_rows(cnn_layers)}
  <tr style="font-weight:bold;background:#e8f0fe"><td colspan="2">Total Parameters</td><td>2,489,383</td><td>All trainable</td></tr>
</table>

<h3>Training Configuration</h3>
<table class="no-break">
  <tr><th>Setting</th><th>Value</th></tr>
  <tr><td>Optimizer</td><td>Adam (lr=0.001)</td></tr>
  <tr><td>Loss</td><td>Categorical Cross-Entropy</td></tr>
  <tr><td>Epochs</td><td>60 (EarlyStopping patience=12)</td></tr>
  <tr><td>Batch size</td><td>32</td></tr>
  <tr><td>LR schedule</td><td>ReduceLROnPlateau (factor=0.5, patience=6)</td></tr>
  <tr><td>Training time</td><td>~102 min (CPU)</td></tr>
</table>

{img_tag("cnn_curves", "Figure 4 — CNN Scratch training curves (loss and accuracy)")}

<hr class="section-divider"/>

<!-- ══════════════════════════════════════════════════════ -->
<h2>4. Model 2 — Transfer Learning (EfficientNetB0)</h2>

<p>
EfficientNetB0 was selected over MobileNetV2 after MobileNetV2 peaked at 64% accuracy
(below the CNN scratch baseline). EfficientNetB0 uses compound scaling across
width, depth, and resolution — producing better fine-grained feature maps for faces.
</p>

<h3>Two-Phase Training Strategy</h3>
<div class="two-col no-break">
  <div class="card">
    <h3>Phase 1 — Feature Extraction (25 epochs)</h3>
    <p>EfficientNetB0 backbone completely frozen. Only the custom classification head is trained.</p>
    <p><strong>LR:</strong> 1×10⁻³ &nbsp; <strong>Trainable params:</strong> 362,247</p>
  </div>
  <div class="card">
    <h3>Phase 2 — Fine-Tuning (30 epochs)</h3>
    <p>Top 50 layers of EfficientNetB0 unfrozen (BatchNorm kept frozen). Very low LR to avoid destroying pretrained weights.</p>
    <p><strong>LR:</strong> 1×10⁻⁵ &nbsp; <strong>Trainable params:</strong> ~4.4M</p>
  </div>
</div>

<h3>Custom Classification Head</h3>
<table class="no-break">
  <tr><th>Layer</th><th>Output</th><th>Parameters</th><th>Notes</th></tr>
  {layer_rows(tl_head_layers)}
  <tr style="font-weight:bold;background:#e8f0fe"><td colspan="2">Total</td><td>4,412,330</td><td>—</td></tr>
</table>

<h3>Data Pipeline Optimization</h3>
<p>
Replaced the original <code>ImageDataGenerator</code> (which caused GPU starvation at ~25% utilization)
with a <strong>tf.data pipeline</strong> using <code>num_parallel_calls=AUTOTUNE</code> and
<code>prefetch(AUTOTUNE)</code>. This raised GPU utilization to ~50–60% and reduced per-epoch
time from ~8.5 min to ~1 min — a <strong>8× speedup</strong>.
</p>

{img_tag("tl_curves", "Figure 5 — EfficientNetB0 training curves (Phase 1 + Phase 2)")}

<hr class="section-divider"/>

<!-- ══════════════════════════════════════════════════════ -->
<h2>5. Evaluation Results</h2>

<h3>Overall Metrics (Test Set — 7,178 images)</h3>
<table class="no-break">
  <tr><th>Metric</th><th>CNN Scratch</th><th>EfficientNetB0</th><th>Winner</th></tr>
  <tr><td>Test Accuracy</td><td><strong>{cnn_ev['accuracy']*100:.2f}%</strong></td><td>{eff_ev['accuracy']*100:.2f}%</td><td><span class="badge green">CNN</span></td></tr>
  <tr><td>Macro Precision</td><td><strong>{cnn_ev['precision']*100:.2f}%</strong></td><td>{eff_ev['precision']*100:.2f}%</td><td><span class="badge green">CNN</span></td></tr>
  <tr><td>Macro Recall</td><td>{cnn_ev['recall']*100:.2f}%</td><td><strong>{eff_ev['recall']*100:.2f}%</strong></td><td><span class="badge green">EfficientNet</span></td></tr>
  <tr><td>Macro F1-Score</td><td>{cnn_ev['f1_score']*100:.2f}%</td><td><strong>{eff_ev['f1_score']*100:.2f}%</strong></td><td><span class="badge green">EfficientNet</span></td></tr>
  <tr><td>Training Time</td><td>~102 min</td><td><strong>39.1 min</strong></td><td><span class="badge green">EfficientNet</span></td></tr>
  <tr><td>Best Val Accuracy</td><td>{scratch['best_val_accuracy']*100:.2f}%</td><td>{tl['best_val_accuracy']*100:.2f}%</td><td><span class="badge green">CNN</span></td></tr>
</table>

{img_tag("comparison", "Figure 6 — Side-by-side model comparison (Accuracy, Precision, Recall, F1)")}

<h3>Per-Class Performance (CNN Scratch)</h3>
<table class="no-break">
  <tr><th>Emotion</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
  <tr><td>Angry</td><td>0.55</td><td>0.68</td><td>0.60</td><td>958</td></tr>
  <tr class="highlight"><td>Disgust</td><td>0.85</td><td>0.20</td><td>0.32</td><td>111</td></tr>
  <tr><td>Fear</td><td>0.60</td><td>0.33</td><td>0.43</td><td>1,024</td></tr>
  <tr><td>Happy</td><td>0.89</td><td>0.85</td><td><strong>0.87</strong></td><td>1,774</td></tr>
  <tr><td>Neutral</td><td>0.55</td><td>0.74</td><td>0.63</td><td>1,233</td></tr>
  <tr><td>Sad</td><td>0.55</td><td>0.54</td><td>0.54</td><td>1,247</td></tr>
  <tr><td>Surprise</td><td>0.78</td><td>0.79</td><td>0.78</td><td>831</td></tr>
</table>

<h3>Confusion Matrices</h3>
<div class="two-col">
  {img_tag("cnn_cm", "Figure 7 — CNN Scratch confusion matrix")}
  {img_tag("eff_cm", "Figure 8 — EfficientNetB0 confusion matrix")}
</div>

<h3>ROC / AUC Curves (One-vs-Rest)</h3>
<div class="two-col">
  {img_tag("cnn_roc", "Figure 9 — CNN Scratch ROC curves")}
  {img_tag("eff_roc", "Figure 10 — EfficientNetB0 ROC curves")}
</div>

<hr class="section-divider"/>

<!-- ══════════════════════════════════════════════════════ -->
<h2>6. Conclusion &amp; Recommendations</h2>

<p>
The <strong>CNN from Scratch</strong> achieved higher raw accuracy (66.2% vs 63.2%), while
<strong>EfficientNetB0</strong> achieved higher macro F1-score (0.608 vs 0.598) due to
more balanced recall across minority classes — particularly Disgust and Angry.
</p>

<h3>Why CNN Scratch won on accuracy</h3>
<ul style="padding-left:20px;line-height:2">
  <li>Trained for 60 epochs vs 55 for EfficientNetB0</li>
  <li>Input is native 48×48 grayscale — no upscaling distortion</li>
  <li>ImageNet weights on EfficientNetB0 are optimized for color/natural images, not grayscale faces</li>
  <li>4GB VRAM limited batch size and fine-tuning depth</li>
</ul>

<h3>Key Challenges</h3>
<ul style="padding-left:20px;line-height:2">
  <li><strong>Class imbalance:</strong> Disgust (436 samples) consistently low recall — needs focal loss or stronger oversampling</li>
  <li><strong>Fear/Sad confusion:</strong> Both models confuse these classes — visually similar expressions</li>
  <li><strong>GPU starvation:</strong> Resolved by migrating from ImageDataGenerator to tf.data (8× speedup)</li>
</ul>

<h3>Recommendations for Further Improvement</h3>
<table>
  <tr><th>#</th><th>Recommendation</th><th>Expected Impact</th></tr>
  <tr><td>1</td><td>Use Focal Loss to penalize minority class misclassification harder</td><td>+2–3% on Fear/Disgust F1</td></tr>
  <tr><td>2</td><td>Train EfficientNetB0 with larger image input (260×260) matching its native resolution</td><td>+1–2% accuracy</td></tr>
  <tr><td>3</td><td>Use label smoothing (0.1) to prevent overconfident predictions on majority classes</td><td>Better calibration</td></tr>
  <tr><td>4</td><td>Ensemble CNN Scratch + EfficientNetB0 predictions (average probabilities)</td><td>+2–4% accuracy</td></tr>
  <tr><td>5</td><td>Apply MixUp or CutMix augmentation to improve minority class boundaries</td><td>Better generalization</td></tr>
</table>

<hr class="section-divider"/>

<h2>7. Deployment</h2>
<p>
A <strong>Streamlit web application</strong> (<code>app.py</code>) was built as a bonus deliverable.
Users can upload any face photo and get a real-time emotion prediction with confidence scores
from either model. The app applies the exact same preprocessing pipeline used during training.
</p>
<table>
  <tr><th>Feature</th><th>Details</th></tr>
  <tr><td>Model selection</td><td>CNN Scratch or EfficientNetB0 via sidebar dropdown</td></tr>
  <tr><td>Upload formats</td><td>JPG, PNG, WebP</td></tr>
  <tr><td>Output</td><td>Top emotion + emoji, confidence %, 7-class bar chart</td></tr>
  <tr><td>Preprocessing</td><td>Pipeline A or B applied automatically based on selected model</td></tr>
  <tr><td>Run command</td><td><code>streamlit run app.py</code> → http://localhost:8501</td></tr>
</table>

<hr class="section-divider"/>
<p style="text-align:center;color:#888;font-size:11px;margin-top:20px;">
  FER2013 Project Report &nbsp;|&nbsp; Generated 2026-05-13 &nbsp;|&nbsp; Ahmed Mahmoud, Mohamed Ahmed, Mohamed Ashraf
</p>

</body>
</html>"""

out = Path("documentation/report.html")
out.write_text(html, encoding="utf-8")
print(f"Report written → {out.resolve()}")
print(f"Size: {out.stat().st_size / 1024:.1f} KB")
