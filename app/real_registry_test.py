import cv2
from insightface.app import FaceAnalysis

from face_registry import FaceRegistry


VIDEO_PATH = "input/video_sample1.mp4"


print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(320, 320)
)

print("InsightFace loaded successfully!")


print("Opening video...")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")


# Read first frame
ret, frame1 = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame.")


# Read another frame
for _ in range(30):
    ret, frame2 = cap.read()

if not ret:
    raise RuntimeError("Could not read second frame.")


print("Generating embeddings...")

faces1 = app.get(frame1)
faces2 = app.get(frame2)


print()
print("==============================")
print("REAL FACE REGISTRY TEST")
print("==============================")

print("Frame 1 faces :", len(faces1))
print("Frame 2 faces :", len(faces2))


if len(faces1) == 0 or len(faces2) == 0:
    print("Not enough faces detected.")
    cap.release()
    exit()


# Use first detected face from each frame
embedding1 = faces1[0].embedding
embedding2 = faces2[0].embedding


registry = FaceRegistry(
    similarity_threshold=0.75
)


# Register first face
face_id, is_new, similarity = registry.identify_or_register(
    embedding1
)

print()
print("First face:")
print("Face ID    :", face_id)
print("New Face   :", is_new)
print("Similarity :", similarity)


# Compare second frame
face_id2, is_new2, similarity2 = registry.identify_or_register(
    embedding2
)

print()
print("Second frame:")
print("Face ID    :", face_id2)
print("New Face   :", is_new2)
print("Similarity :", similarity2)


print()
print("==============================")
print("Test Completed")
print("==============================")
print("Registered Faces :", registry.get_face_count())


cap.release()