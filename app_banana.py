import io
import base64
from typing import List, Dict

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO

# -----------------------
# Inisialisasi model YOLO
# -----------------------
model = YOLO("yolov8n.pt")  # otomatis download sekali

# Cari id kelas "banana" di COCO
CLASS_ID_BANANA = None
for cid, name in model.model.names.items():
    if str(name).lower() == "banana":
        CLASS_ID_BANANA = int(cid)
        break

if CLASS_ID_BANANA is None:
    raise RuntimeError("Kelas 'banana' tidak ditemukan di model YOLO.")

# -----------------------
# Utils
# -----------------------
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def draw_boxes(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """Gambar bounding box pisang ke gambar."""
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['label']} {det['score']:.2f}"
        # kotak kuning
        draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=3)
        # background label
        text_w, text_h = draw.textsize(label)
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=(255, 215, 0))
        draw.text((x1 + 2, y1 - text_h - 2), label, fill=(0, 0, 0))
    return out

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
CORS(app)

@app.get("/banana/health")
def health():
    return {"status": "ok", "banana_class_id": CLASS_ID_BANANA}

@app.post("/banana/detect")
def detect_banana():
    """
    Deteksi buah pisang pada gambar.

    multipart/form-data:
      - image : file gambar (jpg/png)
      - conf  : (opsional) default 0.4
      - iou   : (opsional) default 0.5
    """
    try:
        file = request.files.get("image")
        if not file or file.filename == "":
            return jsonify({"error": "Field 'image' (file gambar) wajib diisi."}), 400

        conf = float(request.form.get("conf", 0.4))
        iou = float(request.form.get("iou", 0.5))

        img = Image.open(file.stream).convert("RGB")
        img_np = np.array(img)

        results = model.predict(
            img_np,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(int)

            for bbox, score, cid in zip(boxes, scores, cls_ids):
                if cid == CLASS_ID_BANANA:
                    x1, y1, x2, y2 = bbox.tolist()
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": float(score),
                        "label": "banana"
                    })

        annotated_b64 = None
        if detections:
            annotated_img = draw_boxes(img, detections)
            annotated_b64 = "data:image/png;base64," + pil_to_base64(annotated_img)

        return jsonify({
            "count": len(detections),
            "detections": detections,
            "image_b64": annotated_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9100, debug=False)
