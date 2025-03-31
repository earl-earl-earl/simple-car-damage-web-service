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

def postprocess_detections(outputs, img_width, img_height, confidence_thres=0.3, iou_thres=0.3):
    outputs = np.transpose(np.squeeze(outputs[0]))
    rows = outputs.shape[0]
    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        class_scores = outputs[i][4:]
        max_score = np.amax(class_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(class_scores)
            x, y, w, h = outputs[i][:4]
            left, top, width, height = int(x - w / 2), int(y - h / 2), int(w), int(h)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, left + width, top + height])

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    detected_labels = []
    if indices is not None:  # Check if any boxes survived NMS
        try:
            if isinstance(indices, tuple):
                # Convert tuple to a NumPy array (handling the case where it might be empty)
                if indices: # Check if the tuple is not empty
                    indices = np.array(indices)
                else:
                    indices = np.array([]) #empty array
            elif not isinstance(indices, np.ndarray):
                # If it's not a tuple or ndarray, try to convert it
                indices = np.array(indices)

            if indices.size > 0:  # Check if the array is not empty.  Prevents error if empty list or array
                indices = indices.flatten() # Flatten regardless

                for i in indices:
                    detected_labels.append(damage_classes.get(str(class_ids[i]], "Unknown"))

        except Exception as e:
            print(f"Error processing indices: {e}")
            return ["Error Processing Detections"]

    return detected_labels if detected_labels else ["No Damage"]

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    img_data = preprocess_image(image, size=(640, 640), swap_channels=True)
    damage_outputs = damage_model.run(None, {damage_model.get_inputs()[0].name: img_data})
    damage_labels = postprocess_detections(damage_outputs, 640, 640)
    
    # Run severity classification
    image.seek(0)  # Reset file pointer
    img_data = preprocess_image(image, size=(224, 224), swap_channels=False)
    severity_outputs = severity_model.run(None, {severity_model.get_inputs()[0].name: img_data})
    severity_prediction = np.argmax(severity_outputs[0])
    severity_label = severity_classes.get(str(severity_prediction), "Unknown")
    
    return jsonify({"damages": list(set(damage_labels)), "severity": severity_label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
