
import os
import platform
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Run ai_model/train.py with Python 3.11 first."
    )

print(f"Loading model from: {MODEL_PATH}", flush=True)
model = load_model(MODEL_PATH)
print("Model loaded.", flush=True)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

_cv2_dir = os.path.dirname(cv2.__file__)
_cascade_path = os.path.join(_cv2_dir, "data", "haarcascade_frontalface_default.xml")
if not os.path.exists(_cascade_path):
    raise FileNotFoundError(f"OpenCV haarcascade not found at {_cascade_path}")

print(f"Loading haarcascade from: {_cascade_path}", flush=True)
face_cascade = cv2.CascadeClassifier(_cascade_path)

print("Opening webcam (index 0)...", flush=True)
if platform.system() == "Windows":
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    video = cv2.VideoCapture(0)
if not video.isOpened():
    raise RuntimeError(
        "Could not open webcam (VideoCapture(0) failed). "
        "Close other apps using the camera, then try again."
    )

WINDOW_NAME = "Emotion Detector"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
try:
    cv2.moveWindow(WINDOW_NAME, 50, 50)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
except Exception:
    pass
print(f"Opened webcam. Press ESC in '{WINDOW_NAME}' window to exit.", flush=True)

while True:
    ret, frame = video.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        prediction = model.predict(face, verbose=0)
        confidence = np.max(prediction)
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()