import os
import json
import time
import cv2
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis

from database import Database
from event_logger import EventLogger
from event_manager import EventManager
from persistent_registry import PersistentFaceRegistry


# ============================================================
# CONFIGURATION
# ============================================================

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    return config


# ============================================================
# MODEL INITIALIZATION
# ============================================================

def initialize_models(config):
    detection_config = config["detection"]
    recognition_config = config["recognition"]

    print()
    print("=" * 70)
    print("INITIALIZING AI MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # YOLO Face Detection
    # --------------------------------------------------------

    model_path = detection_config["model"]

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    print()
    print("[1/2] Loading YOLO face detection model...")
    print(f"      Model: {model_path}")

    yolo_model = YOLO(model_path)

    print("      YOLO model loaded successfully.")

    # --------------------------------------------------------
    # InsightFace / ArcFace
    # --------------------------------------------------------

    recognition_model = recognition_config.get(
        "model",
        "buffalo_l"
    )

    print()
    print("[2/2] Loading InsightFace / ArcFace...")
    print(f"      Model: {recognition_model}")

    insightface_app = FaceAnalysis(
        name=recognition_model,
        providers=["CPUExecutionProvider"]
    )

    insightface_app.prepare(
        ctx_id=-1,
        det_size=(320, 320)
    )

    print("      InsightFace loaded successfully.")

    print()
    print("=" * 70)
    print("AI MODELS READY")
    print("=" * 70)

    return yolo_model, insightface_app


# ============================================================
# FACE CROP
# ============================================================

def extract_face_crop(frame, box):
    if frame is None or frame.size == 0:
        return None

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


# ============================================================
# INSIGHTFACE EMBEDDING
# ============================================================

def get_embedding(insightface_app, face_crop):
    if face_crop is None or face_crop.size == 0:
        return None

    try:
        faces = insightface_app.get(face_crop)

        if not faces:
            return None

        # Select the largest detected face
        face = max(
            faces,
            key=lambda item: (
                item.bbox[2] - item.bbox[0]
            ) * (
                item.bbox[3] - item.bbox[1]
            )
        )

        embedding = face.embedding

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding

    except Exception as error:
        print(
            f"[WARNING] Embedding generation failed: {error}"
        )
        return None


# ============================================================
# DRAW FACE INFORMATION
# ============================================================

def draw_face_annotation(
    frame,
    box,
    face_id,
    track_id,
    confidence=None
):
    x1, y1, x2, y2 = map(int, box)

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Face ID
    # --------------------------------------------------------

    if face_id is None:
        face_text = "Face ID: Pending"
    else:
        face_text = f"Face ID: {face_id}"

    track_text = f"Track ID: {track_id}"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    if confidence is not None:
        confidence_text = f"Confidence: {confidence:.2f}"
    else:
        confidence_text = ""

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text_lines = [
        face_text,
        track_text
    ]

    if confidence_text:
        text_lines.append(confidence_text)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1

    line_height = 22

    text_box_height = (
        len(text_lines) * line_height + 8
    )

    text_box_width = 260

    # Keep text box inside frame
    text_y = max(
        y1 - text_box_height,
        0
    )

    text_x = max(
        x1,
        0
    )

    # Background
    cv2.rectangle(
        frame,
        (
            text_x,
            text_y
        ),
        (
            text_x + text_box_width,
            text_y + text_box_height
        ),
        (0, 0, 0),
        -1
    )

    # Text
    for index, text in enumerate(text_lines):

        baseline_y = (
            text_y
            + 18
            + index * line_height
        )

        cv2.putText(
            frame,
            text,
            (
                text_x + 5,
                baseline_y
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


# ============================================================
# DRAW DASHBOARD
# ============================================================

def draw_dashboard(
    frame,
    frame_number,
    fps,
    unique_visitors,
    active_visitors
):
    height, width = frame.shape[:2]

    dashboard_height = 125

    cv2.rectangle(
        frame,
        (0, 0),
        (
            width,
            dashboard_height
        ),
        (20, 20, 20),
        -1
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "INTELLIGENT FACE TRACKER",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Unique visitors
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Unique Visitors: {unique_visitors}",
        (260, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Active visitors
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Active Visitors: {active_visitors}",
        (260, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Technology
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "YOLO + ByteTrack + InsightFace",
        (width - 420, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "ArcFace Embeddings",
        (width - 300, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# VIDEO SOURCE
# ============================================================

def open_video_source(input_config):
    """
    Select and open either an MP4/video file or an RTSP stream.

    Priority:
        1. input.source
        2. input.rtsp_url
    """

    source = input_config.get(
        "source",
        ""
    )

    rtsp_url = input_config.get(
        "rtsp_url",
        ""
    )

    # Remove accidental whitespace
    if isinstance(source, str):
        source = source.strip()

    if isinstance(rtsp_url, str):
        rtsp_url = rtsp_url.strip()

    # --------------------------------------------------------
    # MP4 / video file has priority
    # --------------------------------------------------------

    if source:
        video_source = source
        source_type = "MP4 / VIDEO FILE"

    # --------------------------------------------------------
    # RTSP fallback
    # --------------------------------------------------------

    elif rtsp_url:
        video_source = rtsp_url
        source_type = "RTSP STREAM"

    else:
        raise ValueError(
            "No video source configured. "
            "Set input.source for MP4/video or "
            "input.rtsp_url for RTSP."
        )

    print()
    print("=" * 70)
    print("VIDEO SOURCE")
    print("=" * 70)

    print(
        f"Source type: {source_type}"
    )

    print(
        f"Source: {video_source}"
    )

    # --------------------------------------------------------
    # Open capture
    # --------------------------------------------------------

    capture = cv2.VideoCapture(
        video_source
    )

    # For RTSP streams, request a reasonable buffer size.
    if source_type == "RTSP STREAM":
        try:
            capture.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )
        except Exception:
            pass

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {video_source}"
        )

    return (
        capture,
        video_source,
        source_type
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    config,
    yolo_model,
    insightface_app,
    registry,
    event_manager
):

    input_config = config["input"]
    detection_config = config["detection"]
    output_config = config["output"]
    testing_config = config.get(
        "testing",
        {}
    )

    # --------------------------------------------------------
    # Open MP4 or RTSP
    # --------------------------------------------------------

    capture, video_source, source_type = (
        open_video_source(input_config)
    )

    print()
    print("=" * 70)
    print("OPENING VIDEO SOURCE")
    print("=" * 70)

    print(
        f"Source type: {source_type}"
    )

    print(
        f"Source: {video_source}"
    )

    # --------------------------------------------------------
    # Video properties
    # --------------------------------------------------------

    fps_input = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps_input <= 0:
        fps_input = 30.0

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    # RTSP streams may report zero properties.
    # Read one frame to determine the actual dimensions.
    first_frame = None

    if (
        frame_width <= 0
        or frame_height <= 0
    ):

        success, first_frame = capture.read()

        if not success:
            capture.release()

            raise RuntimeError(
                f"Could not read first frame from source: "
                f"{video_source}"
            )

        frame_height, frame_width = (
            first_frame.shape[:2]
        )

    print(
        f"Resolution: "
        f"{frame_width}x{frame_height}"
    )

    print(
        f"FPS: {fps_input:.2f}"
    )

    if total_frames > 0:
        print(
            f"Total frames: {total_frames}"
        )
    else:
        print(
            "Total frames: LIVE STREAM"
        )

    # --------------------------------------------------------
    # Output video
    # --------------------------------------------------------

    save_video = output_config.get(
        "save_video",
        True
    )

    display_video = output_config.get(
        "display",
        True
    )

    video_writer = None

    output_video_path = output_config.get(
        "video_path",
        "output/intelligent_face_tracker_output.mp4"
    )

    # --------------------------------------------------------
    # RTSP output handling
    # --------------------------------------------------------

    # For live RTSP streams, saving indefinitely to an MP4
    # may not be desirable. We still respect save_video.
    if save_video:

        output_directory = os.path.dirname(
            output_video_path
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            fps_input,
            (
                frame_width,
                frame_height
            )
        )

        if not video_writer.isOpened():

            capture.release()

            raise RuntimeError(
                "Could not create output video writer."
            )

        print()
        print(
            f"Output video: "
            f"{output_video_path}"
        )

    # --------------------------------------------------------
    # Detection settings
    # --------------------------------------------------------

    confidence = detection_config.get(
        "confidence",
        0.45
    )

    iou = detection_config.get(
        "iou",
        0.5
    )

    img_size = detection_config.get(
        "img_size",
        640
    )

    skip_frames = detection_config.get(
        "skip_frames",
        0
    )

    tracker_name = config[
        "tracking"
    ].get(
        "tracker",
        "bytetrack.yaml"
    )

    max_frames = testing_config.get(
        "max_frames",
        None
    )

    # --------------------------------------------------------
    # Tracking state
    # --------------------------------------------------------

    recognized_faces = {}

    unique_face_ids = set()

    frame_count = 0

    start_time = time.time()

    print()
    print("=" * 70)
    print("STARTING FACE TRACKING")
    print("=" * 70)

    print(
        f"Frame skip: {skip_frames}"
    )

    print(
        f"YOLO confidence: {confidence}"
    )

    print(
        f"YOLO IoU: {iou}"
    )

    print()

    try:

        while True:

            # ------------------------------------------------
            # Read frame
            # ------------------------------------------------

            if first_frame is not None:

                frame = first_frame

                first_frame = None

                success = True

            else:

                success, frame = capture.read()

            if not success:

                # MP4 reaches EOF.
                # RTSP may temporarily fail.
                if source_type == "RTSP STREAM":
                    print(
                        "[WARNING] RTSP frame read failed."
                    )

                break

            frame_count += 1

            # ------------------------------------------------
            # Testing limit
            # ------------------------------------------------

            if (
                max_frames is not None
                and frame_count > max_frames
            ):
                break

            # ------------------------------------------------
            # Detection / tracking
            # ------------------------------------------------

            should_process = (
                frame_count == 1
                or (
                    frame_count
                    % (skip_frames + 1)
                    == 0
                )
            )

            if should_process:

                results = yolo_model.track(
                    frame,
                    persist=True,
                    tracker=tracker_name,
                    conf=confidence,
                    iou=iou,
                    imgsz=img_size,
                    verbose=False
                )

                result = results[0]

                # ------------------------------------------------
                # Boxes
                # ------------------------------------------------

                boxes = result.boxes

                if (
                    boxes is not None
                    and len(boxes) > 0
                ):

                    xyxy = (
                        boxes.xyxy
                        .cpu()
                        .numpy()
                    )

                    if boxes.id is not None:

                        track_ids = (
                            boxes.id
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                    else:

                        track_ids = [
                            None
                        ] * len(xyxy)

                    if boxes.conf is not None:

                        confidences = (
                            boxes.conf
                            .cpu()
                            .numpy()
                        )

                    else:

                        confidences = [
                            None
                        ] * len(xyxy)

                    # ------------------------------------------------
                    # Process each face
                    # ------------------------------------------------

                    for box, track_id, det_conf in zip(
                        xyxy,
                        track_ids,
                        confidences
                    ):

                        if track_id is None:
                            continue

                        track_id = int(
                            track_id
                        )

                        # --------------------------------------------
                        # Face crop
                        # --------------------------------------------

                        face_crop = extract_face_crop(
                            frame,
                            box
                        )

                        if face_crop is None:
                            continue

                        face_id = recognized_faces.get(
                            track_id
                        )

                        # --------------------------------------------
                        # Recognition
                        # --------------------------------------------

                        if face_id is None:

                            embedding = get_embedding(
                                insightface_app,
                                face_crop
                            )

                            if embedding is not None:

                                (
                                    identified_face_id,
                                    is_new,
                                    similarity
                                ) = registry.identify_or_register(
                                    embedding,
                                    track_id=track_id
                                )

                                if (
                                    identified_face_id
                                    is not None
                                ):

                                    face_id = (
                                        identified_face_id
                                    )

                                    recognized_faces[
                                        track_id
                                    ] = face_id

                                    unique_face_ids.add(
                                        face_id
                                    )

                                    # --------------------------------
                                    # Event manager
                                    # --------------------------------

                                    event_manager.process_face(
                                        face_id=face_id,
                                        track_id=track_id,
                                        face_crop=face_crop
                                    )

                        else:

                            unique_face_ids.add(
                                face_id
                            )

                            event_manager.process_face(
                                face_id=face_id,
                                track_id=track_id,
                                face_crop=face_crop
                            )

                        # --------------------------------------------
                        # Draw annotation
                        # --------------------------------------------

                        draw_face_annotation(
                            frame=frame,
                            box=box,
                            face_id=face_id,
                            track_id=track_id,
                            confidence=(
                                float(det_conf)
                                if det_conf is not None
                                else None
                            )
                        )

            # ------------------------------------------------
            # Update missing tracks
            # ------------------------------------------------

            event_manager.update_missing_tracks()

            # ------------------------------------------------
            # Calculate FPS
            # ------------------------------------------------

            elapsed_time = (
                time.time()
                - start_time
            )

            current_fps = (
                frame_count / elapsed_time
                if elapsed_time > 0
                else 0
            )

            # ------------------------------------------------
            # Visitor counts
            # ------------------------------------------------

            unique_visitors = (
                registry.get_face_count()
            )

            active_visitors = (
                event_manager.get_active_count()
            )

            # ------------------------------------------------
            # Dashboard
            # ------------------------------------------------

            draw_dashboard(
                frame=frame,
                frame_number=frame_count,
                fps=current_fps,
                unique_visitors=unique_visitors,
                active_visitors=active_visitors
            )

            # ------------------------------------------------
            # Save output frame
            # ------------------------------------------------

            if video_writer is not None:

                video_writer.write(
                    frame
                )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            if display_video:

                cv2.imshow(
                    "Intelligent Face Tracker",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    print()
                    print(
                        "Q pressed. Stopping..."
                    )

                    break

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                frame_count % 25 == 0
                or frame_count == 1
            ):

                print(
                    f"Frame: {frame_count} | "
                    f"Unique Faces: {unique_visitors} | "
                    f"Active: {active_visitors} | "
                    f"Tracks: {len(recognized_faces)}"
                )

    finally:

        # ----------------------------------------------------
        # Finalize active visitors
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FINALIZING ACTIVE VISITORS")
        print("=" * 70)

        event_manager.finalize_active_visitors()

        # ----------------------------------------------------
        # Release video resources
        # ----------------------------------------------------

        capture.release()

        if video_writer is not None:
            video_writer.release()

        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    elapsed_time = (
        time.time()
        - start_time
    )

    unique_visitors = (
        registry.get_face_count()
    )

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Frames processed: {frame_count}"
    )

    print(
        f"Processing time: "
        f"{elapsed_time:.2f} seconds"
    )

    if elapsed_time > 0:

        print(
            f"Average FPS: "
            f"{frame_count / elapsed_time:.2f}"
        )

    print(
        f"Unique visitors: "
        f"{unique_visitors}"
    )

    print(
        f"Unique Face IDs observed: "
        f"{len(unique_face_ids)}"
    )

    if save_video:

        print(
            f"Output video: "
            f"{output_video_path}"
        )

    print(
        f"Database: "
        f"{config['storage']['database']}"
    )

    print(
        f"Event log: "
        f"{config['storage']['event_log']}"
    )

    print("=" * 70)

    return {
        "frames_processed": frame_count,
        "unique_visitors": unique_visitors,
        "unique_face_ids": len(unique_face_ids),
        "output_video": (
            output_video_path
            if save_video
            else None
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("INTELLIGENT FACE TRACKER")
    print(
        "Auto-Registration + Recognition + "
        "Tracking + Visitor Counting"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    config = load_config()

    # --------------------------------------------------------
    # Create required directories
    # --------------------------------------------------------

    os.makedirs(
        "logs",
        exist_ok=True
    )

    os.makedirs(
        "logs/entries",
        exist_ok=True
    )

    os.makedirs(
        "logs/exits",
        exist_ok=True
    )

    os.makedirs(
        "output",
        exist_ok=True
    )

    os.makedirs(
        "database",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Initialize database
    # --------------------------------------------------------

    database_path = config[
        "storage"
    ][
        "database"
    ]

    database = Database(
        database_path
    )

    print()
    print(
        f"Database: {database_path}"
    )

    # --------------------------------------------------------
    # Event logger
    # --------------------------------------------------------

    event_log_path = config[
        "storage"
    ][
        "event_log"
    ]

    event_logger = EventLogger(
        event_log_path
    )

    print(
        f"Event log: {event_log_path}"
    )

    # --------------------------------------------------------
    # Persistent face registry
    # --------------------------------------------------------

    recognition_config = config[
        "recognition"
    ]

    similarity_threshold = (
        recognition_config.get(
            "similarity_threshold",
            0.75
        )
    )

    max_embeddings_per_face = (
        recognition_config.get(
            "max_embeddings_per_face",
            5
        )
    )

    confirmation_frames = (
        recognition_config.get(
            "confirmation_frames",
            3
        )
    )

    registry = PersistentFaceRegistry(
        database=database,
        event_logger=event_logger,
        similarity_threshold=similarity_threshold,
        max_embeddings_per_face=max_embeddings_per_face,
        confirmation_frames=confirmation_frames
    )

    print(
        f"Registered faces: "
        f"{registry.get_face_count()}"
    )

    # --------------------------------------------------------
    # Event manager
    # --------------------------------------------------------

    max_missing_frames = config[
        "tracking"
    ].get(
        "max_missing_frames",
        30
    )

    event_manager = EventManager(
        database=database,
        event_logger=event_logger,
        max_missing_frames=max_missing_frames
    )

    # --------------------------------------------------------
    # AI models
    # --------------------------------------------------------

    yolo_model, insightface_app = (
        initialize_models(config)
    )

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    try:

        process_video(
            config=config,
            yolo_model=yolo_model,
            insightface_app=insightface_app,
            registry=registry,
            event_manager=event_manager
        )

    except KeyboardInterrupt:

        print()
        print(
            "Processing interrupted by user."
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print(
            f"{type(error).__name__}: {error}"
        )

        raise

    finally:

        # ----------------------------------------------------
        # Close database
        # ----------------------------------------------------

        database.close()

        print()
        print(
            "Database connection closed."
        )

        print()
        print("=" * 70)
        print("APPLICATION FINISHED")
        print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()