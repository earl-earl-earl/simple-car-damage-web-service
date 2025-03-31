from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import json
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins by default

# Load class mappings from JSON files
with open("damage_classes.json", "r") as f:
    damage_classes = json.load(f)

with open("severity_classes.json", "r") as f:
    severity_classes = json.load(f)

# Load ONNX models with optimized settings
damage_model = ort.InferenceSession("car_damage.onnx", providers=["CPUExecutionProvider"])
severity_model = ort.InferenceSession("severity_model.onnx", providers=["CPUExecutionProvider"])

def preprocess_image(image, size, swap_channels=True):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = np.array(img) / 255.0
    
    if swap_channels:
        img = np.transpose(img, (2, 0, 1))  # Swap channels only if needed
    
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(predictions, scores, iou_threshold=0.5):
    if len(predictions) == 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        selected_indices.append(current_idx)

        other_indices = sorted_indices[1:]
        sorted_indices = [
            i for i in other_indices if iou(predictions[current_idx], predictions[i]) < iou_threshold
        ]

    return selected_indices

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    
    # Run damage detection (640x640, Channels First)
    img_data = preprocess_image(image, size=(640, 640), swap_channels=True)
    damage_outputs = damage_model.run(None, {damage_model.get_inputs()[0].name: img_data})
    
    confidence_threshold = 0.3
    damage_scores = damage_outputs[0][0]
    damage_indices = np.where(damage_scores > confidence_threshold)[0]
    
    if len(damage_indices) > 0:
        damage_boxes = [damage_outputs[1][0][i] for i in damage_indices]  # Get bounding boxes
        damage_labels = [damage_classes.get(str(i), "Unknown") for i in damage_indices]
        selected_indices = non_max_suppression(damage_boxes, damage_scores[damage_indices], iou_threshold=0.5)
        damage_labels = [damage_labels[i] for i in selected_indices]
    else:
        damage_labels = ["No Damage"]

    # Run severity classification (224x224, No Channel Swap)
    image.seek(0)  # Reset file pointer
    img_data = preprocess_image(image, size=(224, 224), swap_channels=False)
    severity_outputs = severity_model.run(None, {severity_model.get_inputs()[0].name: img_data})
    severity_prediction = np.argmax(severity_outputs[0])
    severity_label = severity_classes.get(str(severity_prediction), "Unknown")

    return jsonify({"damages": list(set(damage_labels)), "severity": severity_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
