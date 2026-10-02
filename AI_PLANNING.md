# AI Planning

## Intelligent Face Tracker with Auto-Registration and Visitor Counting

### 1. Project Objective

The objective of this project is to build an intelligent face-tracking system that can:

- Detect faces from video or RTSP camera streams.
- Track detected faces across video frames.
- Generate high-quality face embeddings.
- Automatically register previously unknown faces.
- Recognize registered faces when they appear again.
- Assign persistent Face IDs to visitors.
- Record exactly one entry and one exit event for each visit.
- Store cropped face images for entry and exit events.
- Maintain visitor information in a local database.
- Calculate the unique visitor count.
- Maintain detailed application logs for important AI and system events.

---

## 2. AI Technologies Used

### YOLO

YOLO is used as the primary face detection model.

Responsibilities:

- Detect faces in each processed frame.
- Generate bounding boxes around detected faces.
- Provide confidence scores for detections.
- Reduce unnecessary processing by using configurable frame skipping.

Model used:

```text
YOLOv8 Face Detection
models/yolov8n-face-lindevs.pt
```

### InsightFace / ArcFace

InsightFace is used for face recognition and embedding generation.

The system uses:

```text
InsightFace
ArcFace embeddings
buffalo_l model
```

Responsibilities:

- Extract facial features from detected face crops.
- Convert a face into a numerical embedding vector.
- Compare embeddings using cosine similarity.
- Identify previously registered visitors.
- Support automatic registration of new visitors.

The project intentionally uses a modern face embedding approach instead of the older `face_recognition` library.

### ByteTrack

ByteTrack is used for temporary object tracking.

Responsibilities:

- Maintain a temporary tracking ID for a detected face.
- Associate detections across consecutive frames.
- Reduce repeated processing of the same tracked face.
- Detect when a tracked face disappears.

Important distinction:

```text
Track ID = temporary video tracking identity
Face ID  = persistent visitor identity
```

A single visitor can therefore have multiple Track IDs while retaining the same persistent Face ID.

---

## 3. AI Processing Pipeline

```text
Input Video / RTSP
       |
       v
OpenCV Frame Capture
       |
       v
Frame Skipping
       |
       v
YOLO Face Detection
       |
       v
ByteTrack Tracking
       |
       v
Face Crop Extraction
       |
       v
InsightFace / ArcFace
       |
       v
Face Embedding
       |
       v
Cosine Similarity Matching
       |
       +----------------------+
       |                      |
       | Match Found          | No Match
       v                      v
Existing Face ID        Confirmation Frames
                              |
                              v
                         New Face Registration
                              |
                              v
                         Persistent Face ID
       |
       v
Entry / Tracking / Exit Management
       |
       +-------------------+
       |                   |
       v                   v
Face Crop Storage       SQLite Database
       |
       v
events.log
```

---

## 4. Face Recognition Strategy

The system maintains a local face registry.

Each registered visitor contains:

```text
Face ID
   |
   +-- Embedding 1
   +-- Embedding 2
   +-- Embedding 3
   +-- ...
```

When a new embedding is generated:

1. Compare it with registered embeddings.
2. Calculate cosine similarity.
3. If similarity is above the configured threshold, recognize the visitor.
4. Otherwise, temporarily keep the embedding for confirmation.
5. After sufficient confirmation frames, register a new Face ID.

Configuration:

```json
"recognition": {
    "similarity_threshold": 0.75,
    "confirmation_frames": 3,
    "max_embeddings_per_face": 5
}
```

---

## 5. Auto-Registration Strategy

Unknown faces are not immediately registered from one observation.

Instead:

```text
Unknown Face
     |
     v
Temporary Track
     |
     v
Collect Embeddings
     |
     v
Confirmation Frames
     |
     v
Embedding Consistency Check
     |
     +------+
     |      |
   Pass    Fail
     |      |
     v      v
Register   Continue
Face ID    Observation
```

The default configuration requires 3 confirmation frames before a new visitor is registered.

The embeddings from these observations are combined to create a representative embedding for the visitor.

---

## 6. Persistent Face Identity

Temporary tracker IDs are not used as permanent visitor identities.

For example:

```text
Track 12 -> FACE_000001
Track 25 -> FACE_000001
Track 31 -> FACE_000001
```

All three tracking sessions belong to the same visitor.

This allows the system to handle re-identification when the same person disappears and later appears again.

The unique visitor count is based on unique Face IDs and not on the number of Track IDs.

---

## 7. Entry and Exit Detection

The event manager maintains the current active visitors.

When a newly recognized visitor appears:

```text
Face detected
      |
      v
Face recognized/registered
      |
      v
No active visit exists
      |
      v
ENTRY event
```

The system stores:

- Face ID
- Track ID
- Timestamp
- Event type
- Cropped face image

When the visitor disappears for the configured missing-frame period:

```text
Track disappears
      |
      v
Missing-frame counter
      |
      v
Threshold reached
      |
      v
EXIT event
```

The exit event also stores the corresponding face crop.

Configuration:

```json
"tracking": {
    "max_missing_frames": 30
}
```

---

## 8. Database Design

SQLite is used as the local database.

The database contains three main tables.

### `faces`

Stores registered visitor identities.

```text
id
face_id
created_at
```

### `face_embeddings`

Stores embeddings associated with registered Face IDs.

```text
id
face_id
embedding
embedding_dimension
created_at
```

### `events`

Stores visitor events.

```text
id
face_id
event_type
timestamp
track_id
image_path
```

---

## 9. Event Logging

The project maintains a mandatory event log:

```text
logs/events.log
```

Important AI/system events include:

```text
EMBEDDING_GENERATED
REGISTRATION
RECOGNITION
TRACKING
ENTRY
EXIT
```

Example:

```text
EMBEDDING_GENERATED | Face embedding generated
REGISTRATION | New face registered | Face ID: FACE_000001
RECOGNITION | Face recognized | Face ID: FACE_000001 | Similarity: 0.8123
TRACKING | Face tracked | Face ID: FACE_000001 | Track ID: 14
ENTRY | Entry detected | Face ID: FACE_000001 | Track ID: 14
EXIT | Exit detected | Face ID: FACE_000001 | Track ID: 14
```

---

## 10. Configurable AI Parameters

Important AI parameters are stored in `config.json`.

```json
{
    "detection": {
        "confidence": 0.45,
        "iou": 0.5,
        "img_size": 640,
        "skip_frames": 2
    },
    "recognition": {
        "similarity_threshold": 0.75,
        "confirmation_frames": 3,
        "max_embeddings_per_face": 5
    }
}
```

This allows the system to be tuned without modifying the Python source code.

---

## 11. Input Sources

The application supports two input modes.

### MP4 / Video File

Used during development and testing.

```text
input/video_sample1.mp4
```

### RTSP Stream

Designed for interview/demo deployment with an IP camera.

```json
"input": {
    "source": "",
    "rtsp_url": "rtsp://YOUR_CAMERA_ADDRESS"
}
```

Only one source needs to be active at a time.

---

## 12. Unique Visitor Counting

The system counts unique visitors using persistent Face IDs.

Example:

```text
Track IDs:
12
18
27
35

Persistent Face ID:
FACE_000001
FACE_000001
FACE_000001
FACE_000001
```

Unique visitor count:

```text
1
```

Multiple tracking IDs belonging to the same person therefore do not increase the visitor count.

---

## 13. Reliability Measures

Several mechanisms improve reliability:

### Frame Skipping

The detector can skip configurable numbers of frames to reduce CPU usage.

### Confirmation Frames

Unknown faces require multiple observations before registration.

### Embedding Consistency

Multiple embeddings from a temporary track are compared before creating a new identity.

### Persistent Registry

Previously registered faces are loaded from the SQLite database when the application starts.

### Re-identification

A visitor can receive a new temporary Track ID while retaining the same persistent Face ID.

### Event Logging

Important events are recorded in `events.log` for debugging and verification.

---

## 14. CPU Compatibility

The development system can run without an NVIDIA GPU.

The implementation supports CPU inference using:

```text
PyTorch CPU
InsightFace CPUExecutionProvider
```

GPU acceleration can be introduced later for higher throughput.

---

## 15. Development and Production Strategy

### Development

```text
MP4 Video
   |
   v
YOLO + ByteTrack + InsightFace
   |
   v
SQLite + Local Logs + Cropped Images
```

### Interview / Deployment

```text
RTSP Camera
   |
   v
OpenCV
   |
   v
YOLO + ByteTrack + InsightFace
   |
   v
Persistent Face Registry
   |
   v
Database + Event Logs + Images
```

The same processing pipeline is used for both modes. Only the input source changes.

---

## 16. Future Improvements

Possible future enhancements include:

- GPU acceleration.
- Multi-camera support.
- Web dashboard.
- REST API.
- Remote database.
- Real-time visitor analytics.
- Camera-specific visitor statistics.
- Improved re-identification models.
- Automatic database backup.
- Privacy-aware face image retention.
- Docker deployment.
- Cloud deployment.
- Authentication and access control.

---

## 17. AI Development Approach

AI-assisted development was used to accelerate:

- Project architecture planning.
- Module design.
- Debugging.
- Code generation.
- Error analysis.
- Test-case creation.
- Documentation.
- Configuration design.
- System integration.

The implementation was tested locally using the provided MP4 input.

Automated tests verify core registry and database functionality.

---

## 18. Final AI Architecture

```text
                 +----------------------+
                 |   MP4 / RTSP Input   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |      OpenCV          |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   YOLOv8 Face        |
                 |     Detection        |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |      ByteTrack       |
                 |      Tracking        |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 |   Face Crop          |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | InsightFace / ArcFace|
                 | Embedding Generation |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Cosine Similarity    |
                 | Face Registry         |
                 +----------+-----------+
                            |
                +-----------+-----------+
                |                       |
             Existing                 Unknown
              Face                     Face
                |                       |
                v                       v
          Recognition             Confirmation
                                    Frames
                                        |
                                        v
                                  Registration
                                        |
                +-----------------------+
                |
                v
        +-------------------+
        | Persistent Face ID|
        +---------+---------+
                  |
          +-------+-------+
          |               |
          v               v
       ENTRY             EXIT
          |               |
          +-------+-------+
                  |
                  v
       +-----------------------+
       | SQLite + Face Crops   |
       | + events.log          |
       +-----------------------+
```

---

## 19. Conclusion

The proposed AI pipeline combines:

- YOLO for face detection.
- ByteTrack for temporary tracking.
- InsightFace/ArcFace for face embeddings.
- Cosine similarity for identity matching.
- Automatic face registration.
- SQLite for persistent storage.
- OpenCV for video processing.
- Structured image storage for entry and exit evidence.
- Event logging for system traceability.

The architecture separates detection, tracking, recognition, registration, event management, database storage, and logging into independent modules.

This makes the project easier to test, maintain, extend, and deploy.
