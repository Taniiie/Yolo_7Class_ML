import streamlit as st
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# ==================== CONFIG ====================
st.set_page_config(
    page_title="YOLO 7-Class Safety Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== MODERN STYLING ====================
st.markdown("""
<style>
/* Sidebar background */
.sidebar .sidebar-content {
    background-color: #2e3a87;
    color: white;
}

/* Main header */
h1 {
    color: #ff6b6b;
    text-align: center;
}

/* Captions */
h2, h3, p {
    color: #555555;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#ff6b6b,#f06595);
    color: white;
    font-weight: bold;
    border-radius: 8px;
}

/* Metrics cards */
.stMetric {
    border-radius: 10px;
    padding: 10px;
    background: #ffffff;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
}

/* Alerts and events */
.stText {
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if "events" not in st.session_state:
    st.session_state["events"] = []

if "live" not in st.session_state:
    st.session_state["live"] = False

# ==================== HELPERS ====================
def add_event(text):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["events"].insert(0, f"{ts} — {text}")
    st.session_state["events"] = st.session_state["events"][:100]

@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

def boxes_to_df(boxes) -> pd.DataFrame:
    try:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return pd.DataFrame({
            "x1": xyxy[:, 0],
            "y1": xyxy[:, 1],
            "x2": xyxy[:, 2],
            "y2": xyxy[:, 3],
            "confidence": conf,
            "class": cls,
        })
    except:
        return pd.DataFrame()

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Controls")
weights_default = r"D:\Projects\YOLO-7Class-Detection\runs\detect\7class_run\weights\best.pt"
weights_path = st.sidebar.text_input("Model Weights Path", weights_default)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"])
imgsz = st.sidebar.slider("Image Size", 256, 1280, 640, 32)
conf_th = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.01)
iou_th = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.01)
st.sidebar.markdown("---")
source_type = st.sidebar.selectbox("Source", ["Upload Image", "Video File", "Webcam"])
start_btn = st.sidebar.button("Start Webcam" if not st.session_state["live"] else "Stop Webcam")
if start_btn:
    st.session_state["live"] = not st.session_state["live"]

# ==================== LOAD MODEL ====================
if not Path(weights_path).exists():
    st.error(f"❌ Weights not found at: {weights_path}")
    st.stop()

try:
    model = load_model(weights_path)
    add_event("Model loaded successfully.")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ==================== HEADER ====================
st.markdown("<h1>🛰️ YOLO 7-Class Safety Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Detects OxygenTank, FireAlarm, FirstAidBox, and more</p>", unsafe_allow_html=True)

# ==================== METRICS & ALERTS ====================
col1, col2, col3 = st.columns([3, 6, 2])
col3.metric("Device", device.upper())

left, right = st.columns([0.7, 0.3])
feed = left.empty()

with right:
    st.subheader("🔔 Alerts")
    alert_box = st.empty()
    
    st.subheader("📜 Recent Events")
    for e in st.session_state["events"][:10]:
        st.markdown(f"- {e}")
    
    st.subheader("📊 Metrics")
    metrics_col = st.container()

# ==================== DETECTION FUNCTION ====================
def run_detection(pil_image):
    t0 = time.time()
    results = model.predict(pil_image, imgsz=imgsz, conf=conf_th, iou=iou_th, device=device)
    result = results[0]
    plotted = result.plot()[:, :, ::-1]
    df = boxes_to_df(result.boxes)
    t_ms = int((time.time() - t0) * 1000)

    feed.image(plotted, use_container_width=True)

    with metrics_col:
        st.metric("Inference Time", f"{t_ms} ms")
        st.metric("Objects Found", len(df))

    if len(df) == 0:
        alert_box.warning("⚠️ No objects detected.")
    else:
        alert_box.success("✔️ Objects detected")
    return df

# ==================== INPUT SOURCES ====================
if source_type == "Upload Image":
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", width=400)
        run_detection(img)

elif source_type == "Video File":
    vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if vid:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(vid.read())
        cap = cv2.VideoCapture(temp_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            run_detection(pil_img)
        cap.release()

elif source_type == "Webcam":
    cam_image = st.camera_input("Use your webcam")
    if cam_image:
        img = Image.open(cam_image).convert("RGB")
        run_detection(img)
