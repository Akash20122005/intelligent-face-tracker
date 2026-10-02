# Architecture

## Intelligent Face Tracker with Auto-Registration and Visitor Counting

## 1. System Overview

The Intelligent Face Tracker is a modular computer-vision system designed to detect, track, recognize, register, and count visitors from an MP4 video or RTSP camera stream.

The system separates temporary video tracking identity (`Track ID`) from persistent visitor identity (`Face ID`).

### Main Pipeline

```text
MP4 / RTSP Camera
        |
        v
    OpenCV
        |
        v
 Frame Processing
        |
        v
 YOLOv8 Face Detection
        |
        v
     ByteTrack
        |
        v
   Face Crop
        |
        v
InsightFace / ArcFace
        |
        v
 Face Embedding
        |
        v
Cosine Similarity
        |
        +-------------------+
        |                   |
   Existing Face       Unknown Face
        |                   |
        v                   v
 Recognition        Confirmation Frames
                            |
                            v
                      Auto Registration
                            |
                            v
                    Persistent Face ID
                            |
                            v
                 Entry / Exit Management
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
           SQLite       Face Crops    events.log
```

---

## 2. High-Level Architecture

```text
+-------------------------------------------------------------------+
|                    INTELLIGENT FACE TRACKER                      |
+-------------------------------------------------------------------+
|                                                                   |
|  +----------------+       +----------------------+                |
|  | Input Layer    | ----> | Video Processing    |                |
|  | MP4 / RTSP     |       | OpenCV              |                |
|  +----------------+       +----------+-----------+                |
|                                      |                            |
|                                      v                            |
|                         +------------------------+                |
|                         | Face Detection         |                |
|                         | YOLOv8                 |                |
|                         +-----------+------------+                |
|                                     |                             |
|                                     v                             |
|                         +------------------------+                |
|                         | Object Tracking       |                |
|                         | ByteTrack             |                |
|                         +-----------+------------+                |
|                                     |                             |
|                                     v                             |
|                         +------------------------+                |
|                         | Face Recognition       |                |
|                         | InsightFace / ArcFace  |                |
|                         +-----------+------------+                |
|                                     |                             |
|                                     v                             |
|                         +------------------------+                |
|                         | Face Registry          |                |
|                         | Cosine Similarity      |                |
|                         +-----------+------------+                |
|                                     |                             |
|                     +---------------+---------------+             |
|                     |                               |             |
|                     v                               v             |
|              Existing Face                    Unknown Face        |
|                     |                               |             |
|                     |                       Confirmation Frames  |
|                     |                               |             |
|                     |                               v             |
|                     |                         Registration        |
|                     |                               |             |
|                     +---------------+---------------+             |
|                                     |                             |
|                                     v                             |
|                         +------------------------+                |
|                         | Event Manager          |                |
|                         | Entry / Exit / Active  |                |
|                         +-----------+------------+                |
|                                     |                             |
|                  +------------------+------------------+          |
|                  |                  |                  |          |
|                  v                  v                  v          |
|              Database          Face Images        Event Log       |
|              SQLite            Entry/Exit         events.log      |
|                                                                   |
+-------------------------------------------------------------------+
```

---

## 3. Input Layer

The system supports two input modes.

### MP4 Input

Used for development, testing, and demonstration.

```text
input/video_sample1.mp4
```

OpenCV reads frames from the video source.

### RTSP Input

Designed for live camera deployment.

```text
rtsp://YOUR_CAMERA_ADDRESS
```

The application selects the source according to `config.json`.

Example:

```json
"input": {
    "source": "input/video_sample1.mp4",
    "rtsp_url": ""
}
```

For RTSP:

```json
"input": {
    "source": "",
    "rtsp_url": "rtsp://YOUR_CAMERA_ADDRESS"
}
```

Only one input source is expected to be active at a time.

---

## 4. Video Processing Layer

OpenCV is responsible for:

- Opening the video source.
- Reading frames.
- Obtaining resolution and FPS.
- Processing frames sequentially.
- Extracting face crops.
- Drawing annotations and dashboard information.
- Saving the processed output video.
- Displaying frames when enabled.

Frame skipping is configurable through:

```json
"detection": {
    "skip_frames": 2
}
```

This provides a simple CPU-performance control.

---

## 5. Face Detection Layer

YOLOv8 is responsible for detecting faces.

Input:

```text
Video Frame
```

Output:

```text
Bounding Box
Confidence Score
```

Conceptually:

```text
Frame
  |
  v
YOLOv8
  |
  +---- Face 1 -> [x1, y1, x2, y2] + confidence
  |
  +---- Face 2 -> [x1, y1, x2, y2] + confidence
  |
  +---- Face N -> [x1, y1, x2, y2] + confidence
```

Detection parameters are configurable:

```json
"detection": {
    "confidence": 0.45,
    "iou": 0.5,
    "img_size": 640,
    "skip_frames": 2
}
```

---

## 6. Tracking Layer

ByteTrack associates detections across frames.

Example:

```text
Frame 10:
Face detection -> Track ID 15

Frame 11:
Same face -> Track ID 15

Frame 12:
Same face -> Track ID 15
```

If the face temporarily disappears and a new tracker identity is assigned:

```text
Track ID 15 -> Face ID FACE_000004
Track ID 28 -> Face ID FACE_000004
```

Therefore:

```text
Track ID = temporary tracking identity
Face ID  = persistent visitor identity
```

This separation is a core part of the architecture.

---

## 7. Recognition Layer

After a face is detected and tracked, the face crop is passed to InsightFace.

```text
Face Crop
    |
    v
InsightFace
    |
    v
ArcFace
    |
    v
Embedding Vector
```

The embedding is compared with the persistent face registry.

The system uses cosine similarity for comparison.

```text
New Embedding
      |
      v
Compare with stored embeddings
      |
      v
Highest similarity
      |
      +----------------------+
      |                      |
 Above threshold       Below threshold
      |                      |
      v                      v
Recognized Face         Unknown Face
```

The threshold is configurable:

```json
"similarity_threshold": 0.75
```

---

## 8. Face Registry Layer

The face registry maintains persistent visitor identities.

Example:

```text
FACE_000001
    |
    +-- Embedding A
    +-- Embedding B
    +-- Embedding C

FACE_000002
    |
    +-- Embedding A
    +-- Embedding B
```

Each visitor can have multiple embeddings.

Configuration:

```json
"max_embeddings_per_face": 5
```

Maintaining multiple embeddings helps represent appearance variations across frames.

---

## 9. Auto-Registration Layer

When no existing visitor matches the new embedding, the system creates a temporary pending identity.

```text
Unknown Face
     |
     v
Track ID
     |
     v
Collect embeddings
     |
     v
Confirmation frames
     |
     v
Consistency check
     |
     v
Register persistent Face ID
```

Configuration:

```json
"confirmation_frames": 3
```

After confirmation:

```text
Track ID 42
      |
      v
FACE_000030
```

The new identity and representative embedding are stored in the database.

---

## 10. Persistent Registry and Database

At application startup, previously registered embeddings are loaded from SQLite.

```text
SQLite Database
       |
       v
face_embeddings table
       |
       v
Persistent Face Registry
       |
       v
Recognition
```

This allows identities to survive application restarts.

The registry does not depend only on the current video session.

---

## 11. Event Management Layer

The event manager handles visitor state.

Main states:

```text
UNKNOWN
   |
   v
CONFIRMING
   |
   v
REGISTERED / RECOGNIZED
   |
   v
ACTIVE VISIT
   |
   v
EXITED
```

### Entry

A visitor receives an entry event when a persistent Face ID becomes active for a new visit.

```text
Face ID
   |
   v
No active visit
   |
   v
ENTRY
```

### Tracking

While the visitor remains visible:

```text
Face ID + Track ID
       |
       v
TRACKING
```

### Exit

When the current track remains missing for the configured period:

```text
Missing frames
      |
      v
Threshold reached
      |
      v
EXIT
```

Configuration:

```json
"max_missing_frames": 30
```

---

## 12. Event Storage

Each entry and exit event can store:

```text
Face ID
Track ID
Event Type
Timestamp
Image Path
```

Example:

```text
FACE_000004
ENTRY
Track ID: 15
Timestamp: 2026-10-02T18:00:00
Image: logs/entries/2026-10-02/FACE_000004_ENTRY_....jpg
```

Exit images are stored separately:

```text
logs/exits/YYYY-MM-DD/
```

Entry images:

```text
logs/entries/YYYY-MM-DD/
```

---

## 13. Database Architecture

SQLite contains three primary tables.

```text
+----------------------+
| faces                |
+----------------------+
| id                   |
| face_id              |
| created_at            |
+----------+-----------+
           |
           | face_id
           v
+----------------------+
| face_embeddings      |
+----------------------+
| id                   |
| face_id              |
| embedding            |
| embedding_dimension  |
| created_at            |
+----------------------+

+----------------------+
| events               |
+----------------------+
| id                   |
| face_id              |
| event_type           |
| timestamp             |
| track_id              |
| image_path            |
+----------------------+
```

---

## 14. Logging Architecture

Critical system events are written to:

```text
logs/events.log
```

Supported event categories:

```text
EMBEDDING_GENERATED
REGISTRATION
RECOGNITION
TRACKING
ENTRY
EXIT
```

Example flow:

```text
EMBEDDING_GENERATED
        |
        v
RECOGNITION
        |
        v
TRACKING
        |
        v
ENTRY
        |
        v
TRACKING
        |
        v
EXIT
```

The event log provides traceability for the AI pipeline.

---

## 15. Storage Architecture

```text
intelligent-face-tracker/
|
+-- database/
|      |
|      +-- visitors.db
|
+-- logs/
|      |
|      +-- events.log
|      |
|      +-- entries/
|      |     |
|      |     +-- YYYY-MM-DD/
|      |
|      +-- exits/
|            |
|            +-- YYYY-MM-DD/
|
+-- output/
       |
       +-- intelligent_face_tracker_output.mp4
```

This keeps database records, event logs, evidence images, and output videos organized.

---

## 16. Unique Visitor Counting

The system counts unique visitors using persistent Face IDs.

Example:

```text
Track 10 -> FACE_000001
Track 17 -> FACE_000001
Track 31 -> FACE_000001

Track 12 -> FACE_000002
```

Unique visitors:

```text
FACE_000001
FACE_000002
```

Count:

```text
2
```

The number of Track IDs is therefore not used as the visitor count.

---

## 17. Module Architecture

The Python application is divided into modules.

```text
app/
|
+-- config.py
|     Configuration handling
|
+-- detector.py
|     YOLO detection
|
+-- tracker.py
|     ByteTrack integration
|
+-- recognizer.py
|     InsightFace recognition
|
+-- face_registry.py
|     In-memory identity matching
|
+-- persistent_registry.py
|     Database-backed identity registry
|
+-- database.py
|     SQLite operations
|
+-- event_logger.py
|     Critical event logging
|
+-- event_manager.py
|     Entry/exit/active visitor state
|
+-- utils.py
|     File and face-crop utilities
|
+-- main.py
      Main processing pipeline
```

The modular structure makes individual components easier to test and replace.

---

## 18. Runtime Data Flow

For each processed frame:

```text
1. Capture frame
       |
2. Run YOLO detection
       |
3. Update ByteTrack
       |
4. Obtain Track IDs
       |
5. Crop detected faces
       |
6. Generate ArcFace embeddings
       |
7. Search persistent registry
       |
8. Recognize or confirm/register
       |
9. Update event manager
       |
10. Save required events/images
       |
11. Draw annotations/dashboard
       |
12. Write/display output frame
```

---

## 19. Failure Handling

The architecture includes basic resilience mechanisms.

### Missing Input

If the configured source cannot be opened, the application reports the error and stops cleanly.

### Invalid Face Crop

Empty crops are rejected before image storage.

### Unknown Face

An unknown face enters the confirmation pipeline instead of immediately creating a permanent identity.

### Temporary Track Loss

The event manager waits for the configured missing-frame threshold before generating an exit event.

### Application Shutdown

Active visitors are finalized so that their visit can receive an exit event.

---

## 20. Performance Architecture

The system is designed to support CPU execution.

Performance controls include:

- YOLO image size.
- Detection confidence.
- Detection IoU.
- Frame skipping.
- InsightFace detection size.
- ByteTrack tracking.
- Limited embeddings per persistent Face ID.

Example:

```text
skip_frames = 2
```

means the detector can process fewer frames while tracking maintains temporal association.

GPU execution can be added later without changing the overall architecture.

---

## 21. Testing Architecture

Automated unit tests are stored in:

```text
tests/test_core.py
```

The current core test suite verifies:

```text
1. Cosine similarity for identical embeddings
2. Cosine similarity for orthogonal embeddings
3. Face registration
4. Registered face matching
5. Unknown face detection
6. Face database storage
7. Unique visitor count
8. Event storage and retrieval
```

The test command is:

```powershell
python -m unittest discover -s tests -v
```

Expected result:

```text
Ran 8 tests
OK
```

---

## 22. Deployment Architecture

### Local Development

```text
Windows PC
   |
   +-- Python Virtual Environment
   |
   +-- OpenCV
   +-- YOLO
   +-- InsightFace
   +-- SQLite
   |
   +-- MP4 Input
```

### Interview Camera Setup

```text
IP Camera
    |
    | RTSP
    v
Python Application
    |
    +-- YOLO
    +-- ByteTrack
    +-- InsightFace
    +-- Face Registry
    |
    +-- SQLite
    +-- Event Logs
    +-- Entry/Exit Images
```

---

## 23. Security and Privacy Considerations

The system processes biometric face information.

For a production deployment:

- Access to face images should be restricted.
- Database access should be protected.
- Retention periods should be defined.
- Face images should not be exposed publicly.
- Appropriate consent and applicable privacy requirements should be considered.
- Logs should avoid unnecessary sensitive information.

The current hackathon implementation stores data locally for demonstration and testing.

---

## 24. Architecture Summary

The complete architecture separates the problem into five major layers:

```text
+--------------------------------------+
| 1. INPUT                             |
| MP4 / RTSP + OpenCV                 |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| 2. PERCEPTION                       |
| YOLO Face Detection                 |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| 3. TRACKING & RECOGNITION           |
| ByteTrack + InsightFace + ArcFace   |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| 4. IDENTITY & EVENT MANAGEMENT      |
| Face Registry + Entry/Exit Logic    |
+------------------+-------------------+
                   |
                   v
+--------------------------------------+
| 5. STORAGE                          |
| SQLite + Face Crops + events.log    |
+--------------------------------------+
```

This modular architecture provides a clear separation of responsibilities and allows future components such as a web dashboard, API, multi-camera support, GPU inference, or cloud storage to be added without redesigning the complete pipeline.
