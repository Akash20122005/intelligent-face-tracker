import os
from datetime import datetime
import cv2


def create_directories():
    """
    Create all required project directories.
    """

    directories = [
        "logs/entries",
        "logs/exits",
        "output",
        "database"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_face_crop(
    face_crop,
    face_id,
    event_type
):
    """
    Save a cropped face image for ENTRY or EXIT.

    Folder structure:

        logs/
        ├── entries/
        │   └── YYYY-MM-DD/
        │
        └── exits/
            └── YYYY-MM-DD/

    Returns:
        Path of the saved image.
    """

    # Check whether the image is valid
    if face_crop is None or face_crop.size == 0:
        return None

    # Convert event type to uppercase
    event_type = event_type.upper()

    # Only ENTRY and EXIT are allowed
    if event_type not in ["ENTRY", "EXIT"]:
        raise ValueError(
            "event_type must be ENTRY or EXIT"
        )

    # Get today's date
    date_folder = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # Map event type to required folder name
    if event_type == "ENTRY":
        folder_name = "entries"
    else:
        folder_name = "exits"

    # Create date-specific directory
    base_directory = os.path.join(
        "logs",
        folder_name,
        date_folder
    )

    os.makedirs(
        base_directory,
        exist_ok=True
    )

    # Generate unique timestamp
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # Create image filename
    filename = (
        f"{face_id}_{event_type}_{timestamp}.jpg"
    )

    # Complete file path
    filepath = os.path.join(
        base_directory,
        filename
    )

    # Save the face crop
    success = cv2.imwrite(
        filepath,
        face_crop
    )

    if not success:
        raise RuntimeError(
            f"Could not save face crop: {filepath}"
        )

    return filepath