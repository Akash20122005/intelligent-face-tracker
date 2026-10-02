import cv2
from ultralytics import YOLO


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/yolov8n-face-lindevs.pt"
VIDEO_PATH = "input/video_sample1.mp4"
OUTPUT_PATH = "output/tracking_test.mp4"

CONFIDENCE = 0.45
TRACKER = "bytetrack.yaml"


# ==========================================
# LOAD YOLO FACE MODEL
# ==========================================

print("Loading YOLO face model...")

model = YOLO(MODEL_PATH)

print("YOLO face model loaded successfully!")


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

print("Video opened successfully!")


# ==========================================
# VIDEO INFORMATION
# ==========================================

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video resolution: {width} x {height}")
print(f"FPS: {fps}")


# ==========================================
# CREATE OUTPUT VIDEO
# ==========================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)


# ==========================================
# TRACKING VARIABLES
# ==========================================

frame_count = 0

# Stores every unique ByteTrack ID
unique_track_ids = set()


# ==========================================
# PROCESS VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    # End of video
    if not ret:
        break

    frame_count += 1

    # --------------------------------------
    # YOLO DETECTION + BYTETRACK
    # --------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        conf=CONFIDENCE,
        verbose=False
    )

    # --------------------------------------
    # PROCESS RESULTS
    # --------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        # Tracking IDs may not exist
        if boxes.id is None:
            continue

        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            boxes.conf
            .cpu()
            .tolist()
        )

        coordinates = (
            boxes.xyxy
            .int()
            .cpu()
            .tolist()
        )

        # ----------------------------------
        # PROCESS EACH FACE
        # ----------------------------------

        for track_id, confidence, box in zip(
            track_ids,
            confidences,
            coordinates
        ):

            x1, y1, x2, y2 = box

            # Add Track ID to set
            unique_track_ids.add(track_id)

            # ----------------------------------
            # DRAW FACE BOX
            # ----------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ----------------------------------
            # DRAW TRACK ID
            # ----------------------------------

            label = (
                f"Track ID: {track_id} "
                f"Conf: {confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # --------------------------------------
    # DISPLAY FRAME
    # --------------------------------------

    cv2.imshow(
        "YOLO + ByteTrack",
        frame
    )

    # --------------------------------------
    # SAVE FRAME
    # --------------------------------------

    out.write(frame)

    # --------------------------------------
    # PRESS Q TO STOP
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# RELEASE RESOURCES
# ==========================================

cap.release()
out.release()
cv2.destroyAllWindows()


# ==========================================
# FINAL RESULTS
# ==========================================

print()
print("==============================")
print("Tracking Test Completed")
print("==============================")

print(f"Frames processed : {frame_count}")

print(
    f"Unique Track IDs : "
    f"{len(unique_track_ids)}"
)

print(
    f"Track IDs        : "
    f"{sorted(unique_track_ids)}"
)

print(
    f"Output video     : "
    f"{OUTPUT_PATH}"
)