from database import Database
from event_manager import EventManager


DATABASE_PATH = "database/visitors.db"


print("Starting Event Manager Test...")


# Connect to database
db = Database(DATABASE_PATH)


# Create event manager
manager = EventManager(
    database=db,
    max_missing_frames=3
)


print()
print("==============================")
print("TEST 1: New Face Entry")
print("==============================")


# Track 1 appears
manager.process_face(
    face_id="FACE_000001",
    track_id=1,
    image_path="logs/entries/test_entry.jpg"
)


print()
print("Active tracks:", manager.get_active_count())


print()
print("==============================")
print("TEST 2: Same Face Continues")
print("==============================")


# Same track appears again
manager.process_face(
    face_id="FACE_000001",
    track_id=1,
    image_path="logs/entries/test_entry.jpg"
)

manager.process_face(
    face_id="FACE_000001",
    track_id=1,
    image_path="logs/entries/test_entry.jpg"
)


print()
print("Active tracks:", manager.get_active_count())


print()
print("==============================")
print("TEST 3: Face Disappears")
print("==============================")


# Simulate missing frames
manager.update_missing_tracks()
manager.update_missing_tracks()
manager.update_missing_tracks()


print()
print("Active tracks:", manager.get_active_count())


print()
print("==============================")
print("Event Manager Test Completed")
print("==============================")

db.close()