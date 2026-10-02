import os
import cv2
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "input/video_sample1.mp4"
YOLO_MODEL_PATH = "models/yolov8n-face-lindevs.pt"

CONFIDENCE = 0.45
TRACKER = "bytetrack.yaml"

# Process only the first 30 frames
MAX_FRAMES = 30


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(embedding1, embedding2):

    embedding1 = np.asarray(
        embedding1,
        dtype=np.float32
    )

    embedding2 = np.asarray(
        embedding2,
        dtype=np.float32
    )

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(
        np.dot(embedding1, embedding2)
        / (norm1 * norm2)
    )


# ============================================================
# EXTRACT FACE CROP
# ============================================================

def extract_face_crop(frame, box):

    x1, y1, x2, y2 = box

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# GET INSIGHTFACE EMBEDDING
# ============================================================

def get_embedding(
    insightface_app,
    face_crop
):

    if face_crop is None:
        return None

    faces = insightface_app.get(
        face_crop
    )

    if not faces:
        return None

    # Select largest face detected
    face = max(
        faces,
        key=lambda item: (
            item.bbox[2] - item.bbox[0]
        ) * (
            item.bbox[3] - item.bbox[1]
        )
    )

    if face.embedding is None:
        return None

    return face.embedding


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print(" Track-Based Similarity Test")
    print("========================================")

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not os.path.exists(VIDEO_PATH):

        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print()
    print("Loading YOLO...")

    yolo_model = YOLO(
        YOLO_MODEL_PATH
    )

    print(
        "YOLO loaded successfully."
    )

    # --------------------------------------------------------
    # Load InsightFace
    # --------------------------------------------------------

    print()
    print("Loading InsightFace...")

    insightface_app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CPUExecutionProvider"
        ]
    )

    insightface_app.prepare(
        ctx_id=-1,
        det_size=(320, 320)
    )

    print(
        "InsightFace loaded successfully."
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    # --------------------------------------------------------
    # Previous embeddings
    #
    # Dictionary:
    #
    # track_id → previous embedding
    # --------------------------------------------------------

    previous_embeddings = {}

    # Store similarity values
    all_similarities = []

    # Store number of observations
    track_observations = {}

    frame_count = 0

    # --------------------------------------------------------
    # Video processing
    # --------------------------------------------------------

    while frame_count < MAX_FRAMES:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        print(
            f"\nProcessing frame "
            f"{frame_count}/{MAX_FRAMES}..."
        )

        # ----------------------------------------------------
        # YOLO + ByteTrack
        # ----------------------------------------------------

        results = yolo_model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE,
            verbose=False
        )

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            if boxes.id is None:
                continue

            # ------------------------------------------------
            # Track IDs
            # ------------------------------------------------

            track_ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

            # ------------------------------------------------
            # Bounding boxes
            # ------------------------------------------------

            coordinates = (
                boxes.xyxy
                .int()
                .cpu()
                .tolist()
            )

            # ------------------------------------------------
            # Process each tracked face
            # ------------------------------------------------

            for track_id, box in zip(
                track_ids,
                coordinates
            ):

                face_crop = extract_face_crop(
                    frame,
                    box
                )

                embedding = get_embedding(
                    insightface_app,
                    face_crop
                )

                if embedding is None:

                    print(
                        f"Track {track_id}: "
                        f"No embedding"
                    )

                    continue

                # --------------------------------------------
                # First observation
                # --------------------------------------------

                if track_id not in previous_embeddings:

                    previous_embeddings[
                        track_id
                    ] = embedding

                    track_observations[
                        track_id
                    ] = 1

                    print(
                        f"Track {track_id}: "
                        f"First embedding "
                        f"generated"
                    )

                    continue

                # --------------------------------------------
                # Compare with previous embedding
                # --------------------------------------------

                previous_embedding = (
                    previous_embeddings[
                        track_id
                    ]
                )

                similarity = cosine_similarity(
                    previous_embedding,
                    embedding
                )

                # --------------------------------------------
                # Store current embedding
                # --------------------------------------------

                previous_embeddings[
                    track_id
                ] = embedding

                track_observations[
                    track_id
                ] += 1

                all_similarities.append(
                    similarity
                )

                print(
                    f"Track {track_id} | "
                    f"Similarity: "
                    f"{similarity:.4f}"
                )

    # --------------------------------------------------------
    # Close video
    # --------------------------------------------------------

    cap.release()

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" Track Similarity Results")
    print("========================================")

    print(
        f"Frames processed: "
        f"{frame_count}"
    )

    print(
        f"Unique Track IDs: "
        f"{len(track_observations)}"
    )

    print()
    print("Observations per Track:")

    for track_id in sorted(
        track_observations
    ):

        print(
            f"  Track {track_id}: "
            f"{track_observations[track_id]} "
            f"embeddings"
        )

    # --------------------------------------------------------
    # Similarity statistics
    # --------------------------------------------------------

    if not all_similarities:

        print()
        print(
            "No similarity comparisons "
            "were available."
        )

        return

    similarities = np.asarray(
        all_similarities,
        dtype=np.float32
    )

    print()
    print("========================================")
    print(" Similarity Statistics")
    print("========================================")

    print(
        f"Comparisons: "
        f"{len(similarities)}"
    )

    print(
        f"Minimum similarity : "
        f"{similarities.min():.4f}"
    )

    print(
        f"Maximum similarity : "
        f"{similarities.max():.4f}"
    )

    print(
        f"Average similarity : "
        f"{similarities.mean():.4f}"
    )

    print(
        f"Median similarity  : "
        f"{np.median(similarities):.4f}"
    )

    print()
    print("========================================")
    print(" Test Completed")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()