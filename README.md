# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-powered face detection, tracking, recognition, auto-registration, and visitor counting system built with **Python, YOLO, ByteTrack, InsightFace/ArcFace, OpenCV, and SQLite**.

The project supports both **MP4 video input** for development/testing and **RTSP camera streams** for live interview/demo deployment.

---

## 1. Project Overview

The system performs the following operations:

1. Detects faces using YOLOv8.
2. Tracks detected faces using ByteTrack.
3. Generates face embeddings using InsightFace/ArcFace.
4. Matches embeddings using cosine similarity.
5. Automatically registers previously unknown faces.
6. Assigns persistent Face IDs to visitors.
7. Maintains temporary Track IDs for frame-to-frame tracking.
8. Detects visitor entry and exit events.
9. Saves cropped face images for entry and exit events.
10. Stores visitor and event information in SQLite.
11. Writes critical AI/system events to `logs/events.log`.
12. Calculates unique visitor count using persistent Face IDs.

### Identity model

```text
Track ID = temporary video tracking identity
Face ID  = persistent visitor identity
```

A single visitor can therefore have multiple Track IDs while retaining the same Face ID.

---

## 2. Features

### AI Features

- YOLOv8 face detection.
- ByteTrack multi-object tracking.
- InsightFace / ArcFace face embeddings.
- Cosine similarity face matching.
- Automatic unknown-face registration.
- Confirmation-frame based registration.
- Embedding consistency checking.
- Persistent face registry.

### Visitor Management

- Persistent Face IDs.
- Entry event detection.
- Exit event detection.
- Re-identification after temporary tracking loss.
- Unique visitor counting.
- Active visitor tracking.

### Storage

- SQLite database.
- Face embedding storage.
- Entry face crops.
- Exit face crops.
- Structured date-based folders.
- Mandatory application event log.

### Input

- MP4 video.
- RTSP stream support.

### Development

- Configurable detection and recognition parameters.
- CPU-compatible inference.
- Automated unit tests.
- Modular Python architecture.

---

## 3. Architecture

```text
MP4 / RTSP
    |
    v
OpenCV
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
Embedding
    |
    v
Cosine Similarity
    |
    +----------------------+
    |                      |
Existing Face         Unknown Face
    |                      |
    v                      v
Recognition         Confirmation Frames
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

For detailed architecture, see:

```text
ARCHITECTURE.md
```

For AI planning and design decisions, see:

```text
AI_PLANNING.md
```

---

## 4. Project Structure

```text
intelligent-face-tracker/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py
│   ├── tracker.py
│   ├── recognizer.py
│   ├── database.py
│   ├── event_logger.py
│   ├── face_registry.py
│   ├── persistent_registry.py
│   ├── event_manager.py
│   ├── utils.py
│   └── main.py
│
├── models/
│   └── yolov8n-face-lindevs.pt
│
├── input/
│   └── video_sample1.mp4
│
├── logs/
│   ├── entries/
│   ├── exits/
│   └── events.log
│
├── output/
│   └── intelligent_face_tracker_output.mp4
│
├── database/
│   └── visitors.db
│
├── tests/
│   └── test_core.py
│
├── config.json
├── requirements.txt
├── AI_PLANNING.md
├── ARCHITECTURE.md
├── README.md
└── .gitignore
```

---

## 5. Requirements

Recommended environment:

```text
Python 3.12
```

Main libraries:

```text
Ultralytics
OpenCV
NumPy
SciPy
InsightFace
ONNX Runtime
PyTorch
```

The current development setup supports CPU inference.

---

## 6. Installation

### Step 1 — Clone the repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd intelligent-face-tracker
```

### Step 2 — Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
```

### Step 3 — Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4 — Install dependencies

```powershell
pip install -r requirements.txt
```

If required during development:

```powershell
pip install ultralytics opencv-python numpy scipy insightface onnxruntime
```

---

## 7. Configuration

All major runtime settings are stored in:

```text
config.json
```

Example:

```json
{
    "input": {
        "source": "input/video_sample1.mp4",
        "rtsp_url": ""
    },
    "detection": {
        "model": "models/yolov8n-face-lindevs.pt",
        "confidence": 0.45,
        "iou": 0.5,
        "img_size": 640,
        "skip_frames": 2
    },
    "recognition": {
        "model": "buffalo_l",
        "similarity_threshold": 0.75,
        "confirmation_frames": 3,
        "max_embeddings_per_face": 5
    },
    "tracking": {
        "tracker": "bytetrack.yaml",
        "max_missing_frames": 30
    },
    "testing": {
        "max_frames": 240
    },
    "storage": {
        "database": "database/visitors.db",
        "log_directory": "logs",
        "event_log": "logs/events.log"
    },
    "output": {
        "save_video": true,
        "display": true,
        "video_path": "output/intelligent_face_tracker_output.mp4"
    }
}
```

---

## 8. Important Configuration Parameters

### Detection

```json
"confidence": 0.45
```

Minimum YOLO confidence for accepting a face detection.

```json
"iou": 0.5
```

IoU threshold used during detection processing.

```json
"img_size": 640
```

YOLO inference image size.

```json
"skip_frames": 2
```

Controls detector frame skipping and can reduce CPU usage.

---

### Recognition

```json
"similarity_threshold": 0.75
```

Minimum cosine similarity used for recognizing a registered face.

```json
"confirmation_frames": 3
```

Number of observations used before registering an unknown face.

```json
"max_embeddings_per_face": 5
```

Maximum stored embeddings maintained per persistent Face ID.

---

### Tracking

```json
"max_missing_frames": 30
```

Number of missing tracking frames before an exit event is generated.

---

## 9. Running the Application

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Run the application:

```powershell
python -m app.main
```

The application will:

1. Load configuration.
2. Load YOLO.
3. Load InsightFace.
4. Open the configured MP4/RTSP source.
5. Detect faces.
6. Track faces.
7. Generate embeddings.
8. Recognize/register visitors.
9. Generate entry/exit events.
10. Save images and database records.
11. Create an annotated output video.

---

## 10. MP4 Mode

For local testing, use:

```json
"input": {
    "source": "input/video_sample1.mp4",
    "rtsp_url": ""
}
```

Then:

```powershell
python -m app.main
```

The output video is saved to:

```text
output/intelligent_face_tracker_output.mp4
```

---

## 11. RTSP Mode

For an IP camera, set:

```json
"input": {
    "source": "",
    "rtsp_url": "rtsp://YOUR_CAMERA_ADDRESS"
}
```

Example format:

```text
rtsp://username:password@camera-address:554/stream
```

Then run:

```powershell
python -m app.main
```

The same AI processing pipeline is used for the live stream.

---

## 12. Face Recognition and Auto-Registration

The recognition pipeline is:

```text
Face Crop
    |
    v
InsightFace
    |
    v
ArcFace Embedding
    |
    v
Compare with Registered Embeddings
    |
    +----------------------+
    |                      |
   Match                 No Match
    |                      |
    v                      v
Recognize            Confirmation Frames
                           |
                           v
                    Consistency Check
                           |
                           v
                    Register Face ID
```

Example persistent identity:

```text
FACE_000001
```

A visitor may have multiple temporary tracker IDs:

```text
Track 12 -> FACE_000001
Track 28 -> FACE_000001
Track 35 -> FACE_000001
```

The unique visitor count remains:

```text
1
```

---

## 13. Entry and Exit Events

### Entry

When a visitor becomes active:

```text
ENTRY
```

The system stores:

- Face ID
- Track ID
- Timestamp
- Image path

Entry images are stored under:

```text
logs/entries/YYYY-MM-DD/
```

### Exit

When the active track disappears for the configured missing-frame period:

```text
EXIT
```

Exit images are stored under:

```text
logs/exits/YYYY-MM-DD/
```

---

## 14. Event Logging

The mandatory event log is:

```text
logs/events.log
```

Critical events include:

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
2026-10-02 18:00:00 | INFO | REGISTRATION | New face registered | Face ID: FACE_000001
2026-10-02 18:00:01 | INFO | RECOGNITION | Face recognized | Face ID: FACE_000001 | Similarity: 0.8123
2026-10-02 18:00:01 | INFO | TRACKING | Face tracked | Face ID: FACE_000001 | Track ID: 15
2026-10-02 18:00:01 | INFO | ENTRY | Entry detected | Face ID: FACE_000001 | Track ID: 15
2026-10-02 18:00:12 | INFO | EXIT | Exit detected | Face ID: FACE_000001 | Track ID: 15
```

---

## 15. Database

The SQLite database is:

```text
database/visitors.db
```

### `faces`

Stores persistent visitor identities.

```text
id
face_id
created_at
```

### `face_embeddings`

Stores face embeddings.

```text
id
face_id
embedding
embedding_dimension
created_at
```

### `events`

Stores entry and exit events.

```text
id
face_id
event_type
timestamp
track_id
image_path
```

---

## 16. Testing

Automated tests are available in:

```text
tests/test_core.py
```

Run:

```powershell
python -m unittest discover -s tests -v
```

The current test suite contains 8 tests covering:

- Cosine similarity.
- Face registration.
- Registered-face matching.
- Unknown-face handling.
- Database face storage.
- Unique visitor counting.
- Event storage and retrieval.

Expected result:

```text
Ran 8 tests
OK
```

---

## 17. Output

The application generates an annotated video:

```text
output/intelligent_face_tracker_output.mp4
```

The visualization includes:

- Face bounding boxes.
- Face ID.
- Track ID.
- Detection confidence.
- Frame information.
- FPS.
- Unique visitor count.
- Active visitor count.
- Technology information.

Example:

```text
Face ID: FACE_000004
Track ID: 15
Confidence: 0.91
```

---

## 18. Sample Output Structure

```text
logs/
├── events.log
├── entries/
│   └── 2026-10-02/
│       ├── FACE_000001_ENTRY_....jpg
│       └── FACE_000002_ENTRY_....jpg
└── exits/
    └── 2026-10-02/
        ├── FACE_000001_EXIT_....jpg
        └── FACE_000002_EXIT_....jpg
```

Database:

```text
database/
└── visitors.db
```

Video:

```text
output/
└── intelligent_face_tracker_output.mp4
```

---

## 19. Development Validation

The development MP4 was processed successfully using the complete pipeline.

Validated components include:

```text
YOLO face detection             ✓
ByteTrack tracking              ✓
InsightFace recognition         ✓
Persistent Face IDs             ✓
Auto-registration               ✓
SQLite database                 ✓
Entry/exit events               ✓
Face crop storage               ✓
Mandatory event logging         ✓
Annotated output video          ✓
RTSP input support              ✓
Automated core tests            ✓
```

The automated core test suite currently passes:

```text
8 tests
8 passed
```

---

## 20. CPU Execution

The project was developed and tested with CPU inference.

The current setup uses:

```text
PyTorch CPU
InsightFace CPUExecutionProvider
ONNX Runtime
```

A GPU-enabled environment can be used later for higher real-time throughput.

---

## 21. Performance Notes

Performance depends on:

- CPU/GPU hardware.
- YOLO image size.
- Number of faces in a frame.
- Frame skipping.
- InsightFace embedding generation.
- Video resolution.
- RTSP network conditions.

For CPU operation, configurable frame skipping can reduce inference workload.

---

## 22. Assumptions

The implementation assumes:

1. The input video contains visible faces suitable for detection.
2. Face crops are sufficiently clear for embedding generation.
3. A temporary tracking ID can be associated with a persistent Face ID using face embeddings.
4. The configured similarity threshold is appropriate for the deployment environment.
5. Local filesystem and SQLite storage are available.
6. RTSP credentials, if required, are supplied through the configured stream URL.
7. Production deployments should add appropriate privacy, security, and retention controls for biometric data.

---

## 23. Known Limitations

- CPU inference is slower than GPU inference.
- Severe face occlusion can reduce recognition reliability.
- Very small or heavily blurred faces may not generate reliable embeddings.
- Similar-looking individuals can require careful threshold tuning.
- RTSP performance depends on camera/network configuration.
- The current implementation is designed primarily as a hackathon prototype rather than a production biometric security system.

---

## 24. AI Planning

The AI planning document is available here:

```text
AI_PLANNING.md
```

It describes:

- AI model selection.
- Detection strategy.
- Tracking strategy.
- Embedding generation.
- Recognition.
- Auto-registration.
- Confirmation frames.
- Persistent Face IDs.
- Entry/exit logic.
- Database design.
- Event logging.
- CPU compatibility.
- Future improvements.

---

## 25. Architecture Documentation

Detailed system architecture is available here:

```text
ARCHITECTURE.md
```

It documents:

- System overview.
- Input layer.
- Video processing.
- YOLO detection.
- ByteTrack tracking.
- InsightFace recognition.
- Face registry.
- Auto-registration.
- Event management.
- Database architecture.
- Storage architecture.
- Runtime data flow.
- Failure handling.
- Testing architecture.
- Deployment architecture.

---

## 26. Demo Video

Add the final Loom or YouTube demonstration link here:

```text
DEMO VIDEO:
<ADD_LOOM_OR_YOUTUBE_LINK_HERE>
```

The demonstration should show:

1. Project startup.
2. Input video/camera source.
3. Face detection.
4. Track IDs.
5. Face IDs.
6. Recognition.
7. New face registration.
8. Entry event.
9. Exit event.
10. Output video.
11. `events.log`.
12. SQLite database.
13. Unique visitor count.

---

## 27. GitHub Submission Checklist

Before submission, verify:

```text
[ ] README.md
[ ] AI_PLANNING.md
[ ] ARCHITECTURE.md
[ ] requirements.txt
[ ] config.json
[ ] app/
[ ] tests/
[ ] models/
[ ] .gitignore
[ ] Sample input or instructions for obtaining it
[ ] Sample output/log evidence
[ ] Demo video link
[ ] No passwords or private credentials
[ ] No unnecessary generated files
```

Run the final test:

```powershell
python -m unittest discover -s tests -v
```

Then run the application:

```powershell
python -m app.main
```

Verify:

```text
output/intelligent_face_tracker_output.mp4
logs/events.log
database/visitors.db
logs/entries/
logs/exits/
```

---

## 28. AI-Assisted Development

AI tools were used during development for:

- Architecture planning.
- Modular code design.
- Debugging.
- Error analysis.
- Test-case generation.
- Documentation.
- Configuration planning.
- Integration assistance.

The resulting system was tested locally and the core automated tests pass.

---

## 29. Hackathon Bottom Line

This project is a modular AI-based visitor tracking system combining face detection, tracking, recognition, auto-registration, persistent identity management, entry/exit event processing, database storage, and structured logging.

**This project is a part of a hackathon run by https://katomaran.com**
