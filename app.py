"""
app.py
------
Stage 9 (Bonus) — Streamlit web app for FER2013 Facial Expression Recognition.

Features:
  - Upload any face image (jpg, png, webp)
  - Preprocesses via the exact same pipeline used during training
    (grayscale → resize 48x48 → CLAHE → Gaussian blur → normalize)
  - Runs inference on the best saved model (CNN Scratch or Transfer Learning)
  - Shows predicted emotion with confidence bar chart
  - Displays before/after CLAHE preprocessing side-by-side

Run:
    streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FER2013 — Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Emotion config ────────────────────────────────────────────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_EMOJIS = {
    "Angry":    "😠",
    "Disgust":  "🤢",
    "Fear":     "😨",
    "Happy":    "😊",
    "Neutral":  "😐",
    "Sad":      "😢",
    "Surprise": "😲",
}
EMOTION_COLORS = {
    "Angry":    "#E74C3C",
    "Disgust":  "#8E44AD",
    "Fear":     "#2C3E50",
    "Happy":    "#F39C12",
    "Neutral":  "#95A5A6",
    "Sad":      "#2980B9",
    "Surprise": "#27AE60",
}

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "CNN from Scratch (66.3% accuracy)":   "models/cnn_scratch_best.keras",
    "Transfer Learning — EfficientNetB0":  "models/transfer_learning_best.keras",
}


# ── Model loader (cached so it only loads once) ───────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    if not Path(model_path).exists():
        return None
    return tf.keras.models.load_model(model_path)


# ── Preprocessing pipeline (identical to training) ────────────────────────────
def preprocess_for_cnn(img_array: np.ndarray):
    """
    Apply the EXACT same Pipeline A preprocessing used during training.

    Training used ImageDataGenerator(rescale=1.0/255) with NO CLAHE and
    NO Gaussian blur — those were only used in the Stage 2 exploratory
    batch analysis (210 images), NOT the full training pipeline.

    Correct pipeline:
        Grayscale → Resize 48×48 → Normalize [0, 1]

    Returns:
        (model_input, raw_48x48, clahe_48x48)
        model_input   — what the model actually receives (no CLAHE)
        raw_48x48     — resized grayscale for display
        clahe_48x48   — CLAHE version shown for comparison only (NOT fed to model)
    """
    # Convert to grayscale
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    else:
        gray = img_array

    # Resize to 48x48 (same as target_size in ImageDataGenerator)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1] (same as rescale=1/255 in ImageDataGenerator)
    normalized = resized.astype(np.float32) / 255.0
    model_input = normalized.reshape(1, 48, 48, 1)  # add batch + channel dims

    # CLAHE version — for visual comparison display ONLY, not fed to model
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    return model_input, resized, enhanced


def preprocess_for_tl(img_array: np.ndarray) -> np.ndarray:
    """
    Apply Pipeline B preprocessing for EfficientNetB0:
    RGB → Resize 224x224 → EfficientNet preprocess_input
    """
    if len(img_array.shape) == 2:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        rgb = img_array

    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    batch   = resized.astype(np.float32)[np.newaxis, ...]
    batch   = tf.keras.applications.efficientnet.preprocess_input(batch)
    return batch


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0F0E17 0%, #1A1A2E 100%); }
    
    .hero-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, #F39C12, #E74C3C, #8E44AD);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center; color: #B0B0C0; font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid #0F3460; border-radius: 16px;
        padding: 1.5rem; margin: 1rem 0;
    }
    .emotion-badge {
        font-size: 4rem; text-align: center; display: block;
        margin-bottom: 0.5rem;
    }
    .emotion-label {
        font-size: 1.8rem; font-weight: 700;
        text-align: center; margin-bottom: 0.3rem;
    }
    .confidence-text {
        font-size: 1rem; color: #B0B0C0; text-align: center;
    }
    .stProgress > div > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎭 FER2013 Emotion Detector")
    st.markdown("*Facial Expression Recognition*")
    st.divider()

    selected_model_name = st.selectbox(
        "🤖 Select Model",
        options=list(MODEL_PATHS.keys()),
    )
    model_path = MODEL_PATHS[selected_model_name]

    st.divider()
    st.markdown("**📊 Dataset Info**")
    st.markdown("- **28,709** training images")
    st.markdown("- **7,178** test images")
    st.markdown("- **7** emotion classes")
    st.markdown("- **48×48** grayscale")
    st.divider()
    st.markdown("**🏗️ Preprocessing Pipeline**")
    st.markdown("1. Convert to grayscale")
    st.markdown("2. Resize to 48×48")
    st.markdown("3. CLAHE contrast enhancement")
    st.markdown("4. Gaussian blur (denoise)")
    st.markdown("5. Normalize to [0, 1]")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎭 Emotion Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Upload a face photo and the AI will detect the emotion</div>',
    unsafe_allow_html=True
)

# Load model
model = load_model(model_path)
if model is None:
    st.error(
        f"⚠️ Model not found at `{model_path}`.\n\n"
        "Run the training script first:\n"
        "```bash\npython train_scratch.py      # CNN Scratch\n"
        "python train_transfer.py     # Transfer Learning\n```"
    )
    st.stop()

st.success(f"✅ Model loaded: **{selected_model_name}**")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Upload a face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Works best with close-up face photos",
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    # ── Layout: image + result side by side ───────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("#### 📷 Uploaded Image")
        st.image(img_rgb, use_column_width=True, caption="Original image")

    # ── Run inference ─────────────────────────────────────────────────────────
    is_tl = "Transfer" in selected_model_name or "EfficientNet" in selected_model_name

    if is_tl:
        input_tensor = preprocess_for_tl(img_rgb)
        raw_img_show, enh_img_show = None, None
    else:
        input_tensor, raw_img_show, enh_img_show = preprocess_for_cnn(img_rgb)

    probs     = model.predict(input_tensor, verbose=0)[0]
    top_idx   = int(np.argmax(probs))
    top_emotion = EMOTIONS[top_idx]
    top_conf    = float(probs[top_idx])

    with col_result:
        st.markdown("#### 🧠 Prediction")
        color = EMOTION_COLORS[top_emotion]
        emoji = EMOTION_EMOJIS[top_emotion]

        st.markdown(f"""
        <div class="result-card">
            <span class="emotion-badge">{emoji}</span>
            <div class="emotion-label" style="color:{color}">{top_emotion}</div>
            <div class="confidence-text">Confidence: <strong>{top_conf*100:.1f}%</strong></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Full confidence bar chart ──────────────────────────────────────────────
    st.markdown("#### 📊 Confidence per Emotion")
    bar_cols = st.columns(len(EMOTIONS))
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        with bar_cols[i]:
            color = EMOTION_COLORS[emotion]
            is_top = (i == top_idx)
            st.markdown(
                f"<div style='text-align:center;font-size:1.4rem'>{EMOTION_EMOJIS[emotion]}</div>",
                unsafe_allow_html=True
            )
            st.progress(float(prob))
            label_style = f"color:{color};font-weight:{'700' if is_top else '400'}"
            st.markdown(
                f"<div style='text-align:center;font-size:0.75rem;{label_style}'>"
                f"{emotion}<br><strong>{prob*100:.1f}%</strong></div>",
                unsafe_allow_html=True
            )

    # ── Before/After CLAHE (only for CNN scratch) ──────────────────────────────
    if not is_tl and raw_img_show is not None:
        st.divider()
        st.markdown("#### 🔬 Preprocessing Visualization (CLAHE)")
        st.caption("⚠️ The model receives the **raw 48×48 grayscale** image (left). CLAHE is shown for reference only.")
        c1, c2 = st.columns(2)
        with c1:
            st.image(raw_img_show, caption="✅ Model input: resize 48×48 + normalize (no CLAHE)", use_column_width=True, clamp=True)
        with c2:
            st.image(enh_img_show, caption="🔍 CLAHE effect (visual reference only, not fed to model)", use_column_width=True, clamp=True)

else:
    # Empty state placeholder
    st.markdown("---")
    cols = st.columns(7)
    for i, (emotion, emoji) in enumerate(EMOTION_EMOJIS.items()):
        with cols[i]:
            color = EMOTION_COLORS[emotion]
            st.markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:2.5rem'>{emoji}</div>"
                f"<div style='color:{color};font-size:0.8rem;font-weight:600'>{emotion}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.markdown(
        "<div style='text-align:center;color:#B0B0C0;margin-top:1rem'>"
        "Upload a face image above to detect the emotion 👆"
        "</div>",
        unsafe_allow_html=True
    )
