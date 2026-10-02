import numpy as np
import cv2

from utils import create_directories, save_face_crop


print("Starting Utils Test...")


# Create required directories
create_directories()


# Create a fake face image
fake_face = np.zeros(
    (200, 200, 3),
    dtype=np.uint8
)


# Save ENTRY image
entry_path = save_face_crop(
    fake_face,
    "FACE_000001",
    "ENTRY"
)


print()
print("ENTRY image saved:")
print(entry_path)


# Save EXIT image
exit_path = save_face_crop(
    fake_face,
    "FACE_000001",
    "EXIT"
)


print()
print("EXIT image saved:")
print(exit_path)


print()
print("==============================")
print("Utils Test Completed")
print("==============================")