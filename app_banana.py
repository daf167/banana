import io
import base64
from typing import List, Dict
import os

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

# -----------------------
# Load YOLO model from LOCAL FILE
# -----------------------
MODEL_PATH = "yolov8n.pt"   # ganti sesuai nama file lokal kamu

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# Cari ID kelas banana di COCO
CLASS_ID_BANANA = None
for cid, name in model.model.names.items():
    if str(name).lower() == "banana":
        CLASS_ID_BANANA = cid
        break

if CLASS_ID_BANANA is None:
    st.error("❌ Model YOLO tidak memiliki kelas 'banana'")
    st.stop()


# -----------------------
# Drawing utility
# -----------------------
def draw_boxes(image: Image.Image, detections: List[Dict]) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['score']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=3)

        text_w, text_h = draw.textsize(label)
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=(255, 215, 0))
        draw.text((x1 + 3, y1 - text_h - 2), label, fill=(0, 0, 0))

    return out


# -----------------------
# UI
# -----------------------
st.set_page_config(
    page_title="Banana Detection",
    page_icon="🍌",
    layout="wide"
)

st.title("🍌 Banana Detection with YOLOv8 (Local Model)")
st.write("Model YOLO sudah dimuat dari file lokal — tidak download lagi.")

st.sidebar.header("⚙️ Pengaturan Deteksi")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.01)
iou = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.5, 0.01)

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.subheader("📥 Gambar Asli")
    st.image(img, use_column_width=True)

    with st.spinner("🔍 Mendeteksi pisang..."):
        results = model.predict(img_np, conf=conf, iou=iou, verbose=False)[0]

    detections = []

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for bbox, score, cid in zip(boxes, scores, cls_ids):
            if cid == CLASS_ID_BANANA:
                detections.append({
                    "bbox": bbox.tolist(),
                    "score": float(score),
                    "label": "banana"
                })

    st.subheader("📊 Hasil Deteksi")

    if len(detections) == 0:
        st.warning("❌ Tidak ada pisang terdeteksi.")
    else:
        st.success(f"🍌 Pisang terdeteksi: **{len(detections)} buah**")

        annotated = draw_boxes(img, detections)
        st.image(annotated, caption="Hasil Deteksi", use_column_width=True)

        st.json(detections)
else:
    st.info("Silakan upload gambar terlebih dahulu.")
