# app.py
import io
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
import torchvision.transforms as T

# Configuration
MODEL_PATH = "model.onnx"
CLASSES_PATH = "classes.json"
IMG_SIZE = 224

# Load classes
with open(CLASSES_PATH, "r") as f:
    CLASSES = json.load(f)

# Setup ONNX runtime inference session (CPU)
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# preprocessing
preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

app = Flask(__name__)

def predict_image_pil(img_pil):
    x = preprocess(img_pil).unsqueeze(0).numpy().astype(np.float32)
    outputs = sess.run(None, {"input": x})
    logits = outputs[0]  # shape (1, num_classes)
    probs = softmax(logits[0])
    top_idx = int(np.argmax(probs))
    return {"class": CLASSES[top_idx], "confidence": float(probs[top_idx])}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"].read()
    img = Image.open(io.BytesIO(file)).convert("RGB")
    res = predict_image_pil(img)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
