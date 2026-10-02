import cv2
import numpy as np
from insightface.app import FaceAnalysis

print("Loading InsightFace model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("InsightFace model loaded successfully!")

VIDEO_PATH = "input/video_sample1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

print("Video opened successfully!")

ret, frame = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame.")

print("First frame read successfully!")

faces = app.get(frame)

print()
print("==============================")
print("InsightFace Test")
print("==============================")
print(f"Faces detected : {len(faces)}")

for i, face in enumerate(faces):

    print()
    print(f"Face {i + 1}")

    if face.embedding is not None:

        embedding = face.embedding

        print(f"Embedding shape : {embedding.shape}")
        print(f"Embedding size  : {embedding.size}")
        print(f"Embedding type  : {embedding.dtype}")

        print(
            "First 5 values  :",
            embedding[:5]
        )

    else:
        print("Embedding not available.")

cap.release()

print()
print("==============================")
print("Recognition Test Completed")
print("==============================")