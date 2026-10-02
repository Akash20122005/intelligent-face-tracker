import numpy as np

from database import Database
from persistent_registry import PersistentFaceRegistry


DATABASE_PATH = "database/visitors.db"


print("Starting Persistent Registry Test...")


db = Database(DATABASE_PATH)


registry = PersistentFaceRegistry(
    database=db,
    similarity_threshold=0.75
)


print()
print("==============================")
print("Initial Registry")
print("==============================")

print(
    "Registered faces :",
    registry.get_face_count()
)


# Create a test embedding
embedding = np.random.rand(
    512
).astype(np.float32)


# Register / recognize
face_id, is_new, similarity = (
    registry.identify_or_register(
        embedding
    )
)


print()
print("==============================")
print("After Processing")
print("==============================")

print("Face ID     :", face_id)
print("New Face    :", is_new)
print("Similarity  :", similarity)

print(
    "Registered faces :",
    registry.get_face_count()
)


db.close()


print()
print("==============================")
print("Persistent Registry Test Completed")
print("==============================")