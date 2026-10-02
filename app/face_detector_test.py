import cv2
from ultralytics import YOLO


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "models/yolov8n-face-lindevs.pt"
VIDEO_PATH = "input/video_sample1.mp4"
OUTPUT_PATH = "output/face_detection_test.mp4"

CONFIDENCE = 0.45


# -----------------------------
# Load YOLO face model
# -----------------------------
print("Loading YOLO face model...")

model = YOLO(MODEL_PATH)

print("YOLO face model loaded successfully!")


# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

print("Video opened successfully!")


# -----------------------------
# Get video properties
# -----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video resolution: {width} x {height}")
print(f"FPS: {fps}")


# -----------------------------
# Output video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# -----------------------------
# Process video
# -----------------------------
frame_count = 0
total_faces = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Run face detection
    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    # Draw detections
    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"Face {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            total_faces += 1

    # Display frame
    cv2.imshow(
        "YOLO Face Detection Test",
        frame
    )

    # Save frame
    out.write(frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()


print("\n==============================")
print("Face Detection Test Completed")
print("==============================")
print(f"Frames processed : {frame_count}")
print(f"Face detections  : {total_faces}")
print(f"Output video     : {OUTPUT_PATH}")