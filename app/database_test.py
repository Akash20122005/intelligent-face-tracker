import os
import numpy as np

from database import Database


DATABASE_PATH = "database/visitors.db"


print("Starting database test...")


# Remove old test database if it exists
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)


db = Database(DATABASE_PATH)


# Create a fake 512-dimensional embedding
embedding = np.random.rand(512).astype(np.float32)


# Save face
db.save_face(
    "FACE_000001",
    embedding
)


# Read faces
faces = db.get_all_faces()


print()
print("==============================")
print("Database Test")
print("==============================")

print("Faces stored :", len(faces))

for face in faces:

    print()
    print("Face ID     :", face["face_id"])
    print("Dimension   :", face["dimension"])
    print("Embedding shape :", face["embedding"].shape)


# Save event
db.save_event(
    face_id="FACE_000001",
    event_type="ENTRY",
    track_id=1,
    image_path="logs/entries/test.jpg"
)


print()
print("Unique visitors :", db.get_unique_visitor_count())


db.close()


print()
print("==============================")
print("Database Test Completed")
print("==============================")
print("Database :", DATABASE_PATH)