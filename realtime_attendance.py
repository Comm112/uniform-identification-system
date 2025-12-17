# realtime_attendance.py
import cv2
import numpy as np
import onnxruntime as ort
import time
import sqlite3
import json
from PIL import Image
import torchvision.transforms as T

MODEL_PATH = "model.onnx"
CLASSES_PATH = "classes.json"
IMG_SIZE = 224
CONF_THRESHOLD = 0.5  # minimum confidence to consider

# Load classes
with open(CLASSES_PATH, "r") as f:
    CLASSES = json.load(f)

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Setup SQLite
conn = sqlite3.connect("attendance.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    school TEXT,
    timestamp TEXT
)
""")
conn.commit()

# HOG person detector (simple and shipped with OpenCV)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# to avoid repeated logs: store last seen time per school
last_seen = {cls: 0 for cls in CLASSES}
LOG_COOLDOWN = 60  # seconds before logging same class again

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_crop(img_crop_bgr):
    # convert bgr->pil rgb
    img_rgb = cv2.cvtColor(img_crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = preprocess(pil).unsqueeze(0).numpy().astype(np.float32)
    outputs = sess.run(None, {"input": x})
    logits = outputs[0][0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])

def log_attendance(school):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO attendance (school, timestamp) VALUES (?, ?)", (school, now))
    conn.commit()
    print(f"[{now}] Logged attendance for {school}")

def main():
    cap = cv2.VideoCapture(0)  # use default webcam
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detect people (returns rectangles)
        rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        for (x, y, w, h), weight in zip(rects, weights):
            # optionally filter by weight threshold
            # draw rectangle
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

            # crop with small margin
            pad = 10
            x0 = max(0, x-pad)
            y0 = max(0, y-pad)
            x1 = min(frame.shape[1], x+w+pad)
            y1 = min(frame.shape[0], y+h+pad)
            crop = frame[y0:y1, x0:x1]

            try:
                school, conf = predict_crop(crop)
            except Exception as e:
                print("Prediction error:", e)
                continue

            label = f"{school} {conf:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # log attendance if confidence high and cooldown passed
            now = time.time()
            if conf >= CONF_THRESHOLD and (now - last_seen[school] > LOG_COOLDOWN):
                log_attendance(school)
                last_seen[school] = now

        cv2.imshow("Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()
