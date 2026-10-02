import numpy as np
from face_registry import FaceRegistry


print("Starting Face Registry Test...")

registry = FaceRegistry(similarity_threshold=0.75)


# Create a fake 512-dimensional embedding
embedding1 = np.random.rand(512).astype(np.float32)

# Same embedding
embedding2 = embedding1.copy()

# Completely different embedding
embedding3 = np.random.rand(512).astype(np.float32)


# -----------------------------
# Test 1: Register first face
# -----------------------------

face_id, is_new, similarity = registry.identify_or_register(embedding1)

print()
print("Test 1")
print("Face ID     :", face_id)
print("New Face    :", is_new)
print("Similarity  :", similarity)


# -----------------------------
# Test 2: Same face
# -----------------------------

face_id, is_new, similarity = registry.identify_or_register(embedding2)

print()
print("Test 2")
print("Face ID     :", face_id)
print("New Face    :", is_new)
print("Similarity  :", similarity)


# -----------------------------
# Test 3: Different face
# -----------------------------

face_id, is_new, similarity = registry.identify_or_register(embedding3)

print()
print("Test 3")
print("Face ID     :", face_id)
print("New Face    :", is_new)
print("Similarity  :", similarity)


# -----------------------------
# Final count
# -----------------------------

print()
print("==============================")
print("Registry Test Completed")
print("==============================")
print("Registered Faces :", registry.get_face_count())