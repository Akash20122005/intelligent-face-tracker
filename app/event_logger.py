import logging
import os
from datetime import datetime


class EventLogger:
    """
    Handles application event logging.

    All critical system events are stored in:
        logs/events.log
    """

    def __init__(self, log_file="logs/events.log"):

        self.log_file = log_file

        # Create the logs directory if it doesn't exist
        os.makedirs(
            os.path.dirname(log_file),
            exist_ok=True
        )

        # Create logger
        self.logger = logging.getLogger(
            "IntelligentFaceTracker"
        )

        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:

            file_handler = logging.FileHandler(
                log_file,
                encoding="utf-8"
            )

            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )

            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    def log(
        self,
        event_type,
        message
    ):
        """
        Write a general event to the log.
        """

        self.logger.info(
            f"{event_type} | {message}"
        )

    def embedding_generated(
        self,
        face_id=None
    ):
        """
        Log embedding generation.
        """

        message = "Face embedding generated"

        if face_id:
            message += f" | Face ID: {face_id}"

        self.log(
            "EMBEDDING_GENERATED",
            message
        )

    def registration(
        self,
        face_id
    ):
        """
        Log new face registration.
        """

        self.log(
            "REGISTRATION",
            f"New face registered | Face ID: {face_id}"
        )

    def recognition(
        self,
        face_id,
        similarity
    ):
        """
        Log successful face recognition.
        """

        self.log(
            "RECOGNITION",
            f"Face recognized | "
            f"Face ID: {face_id} | "
            f"Similarity: {similarity:.4f}"
        )

    def tracking(
        self,
        face_id,
        track_id
    ):
        """
        Log tracking information.
        """

        self.log(
            "TRACKING",
            f"Face tracked | "
            f"Face ID: {face_id} | "
            f"Track ID: {track_id}"
        )

    def entry(
        self,
        face_id,
        track_id,
        image_path=None
    ):
        """
        Log ENTRY event.
        """

        message = (
            f"Entry detected | "
            f"Face ID: {face_id} | "
            f"Track ID: {track_id}"
        )

        if image_path:
            message += f" | Image: {image_path}"

        self.log(
            "ENTRY",
            message
        )

    def exit(
        self,
        face_id,
        track_id,
        image_path=None
    ):
        """
        Log EXIT event.
        """

        message = (
            f"Exit detected | "
            f"Face ID: {face_id} | "
            f"Track ID: {track_id}"
        )

        if image_path:
            message += f" | Image: {image_path}"

        self.log(
            "EXIT",
            message
        )