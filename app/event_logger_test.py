from event_logger import EventLogger


print("Starting Event Logger Test...")


logger = EventLogger(
    "logs/events.log"
)


# Test embedding generation
logger.embedding_generated(
    face_id="FACE_000001"
)


# Test registration
logger.registration(
    face_id="FACE_000001"
)


# Test recognition
logger.recognition(
    face_id="FACE_000001",
    similarity=0.91
)


# Test tracking
logger.tracking(
    face_id="FACE_000001",
    track_id=5
)


# Test entry
logger.entry(
    face_id="FACE_000001",
    track_id=5,
    image_path="logs/entries/2026-10-01/test.jpg"
)


# Test exit
logger.exit(
    face_id="FACE_000001",
    track_id=5,
    image_path="logs/exits/2026-10-01/test.jpg"
)


print()
print("==============================")
print("Event Logger Test Completed")
print("==============================")
print("Log file: logs/events.log")