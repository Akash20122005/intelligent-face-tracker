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

# Compare every 3 frames
FRAME_INTERVAL = 3

# Number of frames to test
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
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print(" Face Embedding Similarity Test")
    print("========================================")

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print()
    print("Loading YOLO...")

    yolo_model = YOLO(
        YOLO_MODEL_PATH
    )

    print("YOLO loaded successfully.")

    # --------------------------------------------------------
    # Load InsightFace
    # --------------------------------------------------------

    print()
    print("Loading InsightFace...")

    insightface_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    insightface_app.prepare(
        ctx_id=-1,
        det_size=(320, 320)
    )

    print("InsightFace loaded successfully.")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    if not os.path.exists(VIDEO_PATH):

        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    # --------------------------------------------------------
    # Store embeddings
    # --------------------------------------------------------

    embeddings = []

    frame_numbers = []

    frame_count = 0

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    while frame_count < MAX_FRAMES:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Only process selected frames
        if frame_count % FRAME_INTERVAL != 0:
            continue

        print(
            f"\nProcessing frame {frame_count}..."
        )

        # ----------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------

        results = yolo_model(
            frame,
            conf=CONFIDENCE,
            verbose=False
        )

        found_embedding = False

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            if len(boxes) == 0:
                continue

            # ------------------------------------------------
            # Select the largest detected face
            # ------------------------------------------------

            largest_area = 0
            largest_box = None

            for box_tensor in boxes.xyxy:

                x1, y1, x2, y2 = (
                    box_tensor
                    .int()
                    .cpu()
                    .tolist()
                )

                area = (
                    max(0, x2 - x1)
                    * max(0, y2 - y1)
                )

                if area > largest_area:

                    largest_area = area
                    largest_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

            if largest_box is None:
                continue

            x1, y1, x2, y2 = largest_box

            # ------------------------------------------------
            # Crop face
            # ------------------------------------------------

            height, width = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            face_crop = frame[
                y1:y2,
                x1:x2
            ]

            if face_crop.size == 0:
                continue

            # ------------------------------------------------
            # Generate InsightFace embedding
            # ------------------------------------------------

            faces = insightface_app.get(
                face_crop
            )

            if not faces:
                continue

            face = max(
                faces,
                key=lambda item: (
                    item.bbox[2] - item.bbox[0]
                ) * (
                    item.bbox[3] - item.bbox[1]
                )
            )

            if face.embedding is None:
                continue

            embedding = face.embedding

            embeddings.append(
                embedding
            )

            frame_numbers.append(
                frame_count
            )

            print(
                f"Embedding generated."
            )

            print(
                f"Embedding dimension: "
                f"{len(embedding)}"
            )

            found_embedding = True

            break

        if not found_embedding:

            print(
                "No usable face embedding found."
            )

    # --------------------------------------------------------
    # Close video
    # --------------------------------------------------------

    cap.release()

    # --------------------------------------------------------
    # Similarity analysis
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" Similarity Results")
    print("========================================")

    print(
        f"Embeddings collected: "
        f"{len(embeddings)}"
    )

    if len(embeddings) < 2:

        print(
            "Not enough embeddings for comparison."
        )

        return

    similarities = []

    for i in range(
        1,
        len(embeddings)
    ):

        similarity = cosine_similarity(
            embeddings[i - 1],
            embeddings[i]
        )

        similarities.append(
            similarity
        )

        print(
            f"Frame "
            f"{frame_numbers[i - 1]} → "
            f"Frame "
            f"{frame_numbers[i]} : "
            f"{similarity:.4f}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    similarities = np.asarray(
        similarities,
        dtype=np.float32
    )

    print()
    print("========================================")
    print(" Similarity Statistics")
    print("========================================")

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
    print("Test completed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()