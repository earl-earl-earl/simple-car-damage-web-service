from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import cv2
import json
import os

app = Flask(__name__)

# Load class mappings from JSON files
with open("damage_classes.json", "r") as f:
    damage_classes = json.load(f)

with open("severity_classes.json", "r") as f:
    severity_classes = json.load(f)

# Load ONNX models with optimized settings
damage_model = ort.InferenceSession("car_damage.onnx.onnx", providers=["CPUExecutionProvider"])
severity_model = ort.InferenceSession("severity_model.onnx", providers=["CPUExecutionProvider"])


def preprocess_image(image, size):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    # Run damage detection
    img_data = preprocess_image(image, size=(640, 640))
    damage_outputs = damage_model.run(None, {damage_model.get_inputs()[0].name: img_data})

    confidence_threshold = 0.3
    damage_scores = damage_outputs[0][0]
    damage_indices = np.where(damage_scores > confidence_threshold)[0]
    damage_labels = [damage_classes.get(str(i), "Unknown") for i in damage_indices]
    damage_labels = damage_labels if damage_labels else ["No Damage"]

    # Run severity classification
    image.seek(0)  # Reset file pointer
    img_data = preprocess_image(image, size=(224, 224))
    severity_outputs = severity_model.run(None, {severity_model.get_inputs()[0].name: img_data})
    severity_prediction = np.argmax(severity_outputs[0])
    severity_label = severity_classes.get(str(severity_prediction), "Unknown")

    return jsonify({"damages": damage_labels, "severity": severity_label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
