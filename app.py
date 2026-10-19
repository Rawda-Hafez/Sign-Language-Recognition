import os
import json
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras

# ================== Page Config ==================
st.set_page_config(
    page_title="ASL Sign Language Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Custom CSS Styling ==================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
    }
    .prediction-letter {
        font-size: 4rem;
        font-weight: 900;
        line-height: 1;
        margin: 5px 0;
    }
    .badge-info {
        background-color: #f0f2f6;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ================== Load Model & Labels ==================
@st.cache_resource
def load_model_and_labels():
    model = None
    model_name = ""
    # Search for available model files in priority order
    candidates = [
        "sign_mnist_finetuned.keras",
        "sign_mnist_finetuned.h5",
        "sign_mnist_cnn.h5"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                model = keras.models.load_model(path, compile=False)
                model_name = path
                break
            except Exception:
                continue

    # Load label map
    label_map = {}
    if os.path.exists("label_map.json"):
        with open("label_map.json", "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
            label_map = {int(k): v for k, v in raw_labels.items()}
    else:
        # Fallback 24 classes (A-Y excluding J and Z)
        letters = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) not in ['J', 'Z']]
        label_map = {i: letter for i, letter in enumerate(letters)}

    return model, model_name, label_map

model, model_path, label_map = load_model_and_labels()

# ================== Sidebar ==================
with st.sidebar:
    st.header("⚙️ Settings & Info")
    
    if model is not None:
        st.success(f"Loaded: `{model_path}`")
    else:
        st.error("❌ Model weights not found!")

    confidence_threshold = st.slider(
        "Confidence Threshold (%)",
        min_value=30,
        max_value=95,
        value=60,
        step=5
    ) / 100.0

    st.markdown("---")
    st.markdown("### ℹ️ Supported Letters (24 Signs)")
    st.markdown(
        "Signs for **J** and **Z** are excluded because they require dynamic motion gestures in ASL."
    )
    
    badges_html = "".join([f"<span class='badge-info'>{v}</span>" for v in label_map.values()])
    st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Ensure good lighting and keep hand inside the green region for best accuracy.")

# ================== Main Header ==================
st.markdown("<div class='main-header'>🤟 ASL Sign Language Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Real-time American Sign Language alphabet recognition using Deep CNN</div>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Please train or place a valid model file (`sign_mnist_finetuned.keras` or `sign_mnist_finetuned.h5`) in the project directory.")
    st.stop()

# ================== Helper Functions ==================
def preprocess_roi(roi_img):
    """Convert input ROI (BGR) to normalized 28x28 grayscale image for CNN input."""
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    tensor = norm.reshape(1, 28, 28, 1)
    return tensor, resized

def predict_hand(roi_img):
    """Predict top class and top-3 probabilities."""
    tensor, preview_28x28 = preprocess_roi(roi_img)
    probs = model.predict(tensor, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    
    best_idx = top_indices[0]
    best_conf = float(probs[best_idx])
    best_label = label_map.get(best_idx, f"Class {best_idx}")
    
    top3 = [(label_map.get(i, f"Class {i}"), float(probs[i])) for i in top_indices]
    return best_label, best_conf, top3, preview_28x28

# ================== Mode Selection ==================
mode = st.radio(
    "Choose Input Method:",
    ["🎥 Live Webcam Stream", "📸 Camera Snapshot", "🖼️ Upload Image"],
    horizontal=True
)

# =======================================================
# Mode 1: Live Webcam Stream
# =======================================================
if mode == "🎥 Live Webcam Stream":
    st.subheader("Live Webcam Detection with Hand Bounding Box")
    st.caption("Place your hand inside the green box. Only the boxed area is analyzed by the AI model.")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("#### Controls")
        if "webcam_active" not in st.session_state:
            st.session_state.webcam_active = False

        c_start, c_stop = st.columns(2)
        if c_start.button("▶️ Start Camera", use_container_width=True, type="primary"):
            st.session_state.webcam_active = True
        if c_stop.button("⏹️ Stop Camera", use_container_width=True):
            st.session_state.webcam_active = False

        pred_placeholder = st.empty()
        preview_placeholder = st.empty()
        chart_placeholder = st.empty()

    with col1:
        video_placeholder = st.empty()

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("⚠️ Unable to access webcam. Please make sure no other program is using it.")
            st.session_state.webcam_active = False
        else:
            box_size = 240
            frame_counter = 0
            
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠️ Failed to capture video frame.")
                    break

                # Flip horizontally for natural mirror feel
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Position bounding box (centered horizontally or right-aligned)
                x1 = int((w - box_size) / 2)
                y1 = int((h - box_size) / 2)
                x2 = x1 + box_size
                y2 = y1 + box_size

                # Extract Region of Interest (ROI)
                roi = frame[y1:y2, x1:x2]

                # Draw ROI box on display frame
                display_frame = frame.copy()
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display_frame,
                    "Place Hand Here",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Predict every 2 frames for smooth performance
                frame_counter += 1
                if frame_counter % 2 == 0 and roi.size > 0:
                    best_label, best_conf, top3, preview_img = predict_hand(roi)

                    # Update Prediction Card
                    if best_conf >= confidence_threshold:
                        pred_placeholder.markdown(f"""
                        <div class='prediction-card'>
                            <div>Detected Sign</div>
                            <div class='prediction-letter'>{best_label}</div>
                            <div>Confidence: <b>{best_conf * 100:.1f}%</b></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        pred_placeholder.warning(f"⚠️ Low confidence ({best_conf * 100:.1f}%). Hold hand clearly inside the green box.")

                    # Show 28x28 preview
                    with preview_placeholder.container():
                        st.caption("AI Model Input (28x28):")
                        st.image(preview_img, width=100)

                    # Show Top-3 Probabilities
                    with chart_placeholder.container():
                        st.caption("Top-3 Predictions:")
                        for letter, prob in top3:
                            st.progress(prob, text=f"Letter {letter}: {prob * 100:.1f}%")

                # Show main camera stream
                rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_display, channels="RGB", use_column_width=True)

            cap.release()
            video_placeholder.info("Camera turned off.")

# =======================================================
# Mode 2: Camera Snapshot
# =======================================================
elif mode == "📸 Camera Snapshot":
    st.subheader("Take a Snapshot")
    st.caption("Take a photo of your hand gesture using your browser's camera.")

    col1, col2 = st.columns([1, 1])
    with col1:
        camera_img = st.camera_input("Capture Hand Gesture")

    if camera_img is not None:
        image = Image.open(camera_img)
        img_np = np.array(image)
        # Convert RGB to BGR for cv2
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        h, w = img_bgr.shape[:2]
        crop_size = min(h, w)
        # Center crop hand
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped_hand = img_bgr[start_y:start_y + crop_size, start_x:start_x + crop_size]

        best_label, best_conf, top3, preview_img = predict_hand(cropped_hand)

        with col2:
            st.markdown("#### Recognition Result")
            if best_conf >= confidence_threshold:
                st.markdown(f"""
                <div class='prediction-card'>
                    <div>Detected Sign</div>
                    <div class='prediction-letter'>{best_label}</div>
                    <div>Confidence: <b>{best_conf * 100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Low confidence ({best_conf * 100:.1f}%). Try with clearer lighting or positioning.")

            st.caption("Preprocessed Image sent to Model (28x28):")
            st.image(preview_img, width=120)

            st.caption("Top-3 Predictions:")
            for letter, prob in top3:
                st.progress(prob, text=f"Letter {letter}: {prob * 100:.1f}%")

# =======================================================
# Mode 3: Upload Image
# =======================================================
elif mode == "🖼️ Upload Image":
    st.subheader("Upload an Image File")
    st.caption("Upload an image of a hand sign (JPG, JPEG, PNG).")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if len(img_np.shape) == 3 and img_np.shape[2] == 3 else img_np

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        best_label, best_conf, top3, preview_img = predict_hand(img_bgr)

        with col2:
            st.markdown("#### Recognition Result")
            if best_conf >= confidence_threshold:
                st.markdown(f"""
                <div class='prediction-card'>
                    <div>Detected Sign</div>
                    <div class='prediction-letter'>{best_label}</div>
                    <div>Confidence: <b>{best_conf * 100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Low confidence ({best_conf * 100:.1f}%). Image might contain too much background noise.")

            st.caption("Model Input (28x28):")
            st.image(preview_img, width=120)

            st.caption("Top-3 Predictions:")
            for letter, prob in top3:
                st.progress(prob, text=f"Letter {letter}: {prob * 100:.1f}%")
