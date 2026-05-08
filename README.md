# Distributed Real-Time Smart City Traffic Command System

A decoupled, two-process distributed system for real-time vehicle detection, tracking, and intelligent traffic command. A Python edge node runs a computer vision pipeline at 23 FPS, broadcasting live telemetry over TCP to a multithreaded Java Command Center that visualizes data, logs it to a database, and provides interactive emergency routing controls.

**Course:** Advanced Programming Design | **University:** Wuhan Institute of Technology

---

## System Architecture

```
┌─────────────────────────────────────────────┐        TCP Socket       ┌────────────────────────────────────────────────┐
│           PYTHON EDGE NODE                  │  ─── JSON @ 1s ──────► │           JAVA COMMAND CENTER                  │
│                                             │       port 9999         │                                                │
│  Video File                                 │                         │  ServerSocket (background thread)              │
│      │                                      │                         │      │                                         │
│      ▼                                      │                         │      ▼                                         │
│  OpenCV Frame Reader                        │                         │  JSON Parser → Live State                      │
│      │ (every other frame skipped)          │                         │      │                                         │
│      ▼                                      │                         │      ├── JavaFX GUI (vehicle count,            │
│  YOLOv11n ONNX Inference (3.2ms/frame)     │                         │      │    congestion badge, class bars)        │
│      │ [1, 3, 640, 640] → [1, 12, 8400]    │                         │      │                                         │
│      ▼                                      │                         │      ├── JFreeChart (live time-series graph)   │
│  Confidence Bias Correction                 │                         │      │                                         │
│  + NMS (IoU=0.45, conf=0.40)               │                         │      ├── SQLite via JDBC (background thread)   │
│      │                                      │                         │      │                                         │
│      ▼                                      │                         │      └── Emergency Routing Controls            │
│  ByteTrack (lost_buffer=30)                 │                         │           (ambulance green wave, etc.)         │
│      │ Persistent unique IDs per vehicle    │                         │                                                │
│      ▼                                      │                         └────────────────────────────────────────────────┘
│  Congestion Calculator                      │
│  (LOW / MEDIUM / HIGH)                      │
│      │                                      │
│      ▼                                      │
│  TCP Broadcaster (background thread)        │
│  + OpenCV Annotated Display Window          │
└─────────────────────────────────────────────┘
```

---

## AI Model Performance

The YOLOv11n model was trained on a combined dataset of 11,218 annotated traffic images using a Kaggle T4 GPU.

| Metric | Value |
|--------|-------|
| mAP@50 | **91.4%** |
| mAP@50-95 | **70.4%** |
| Inference speed | **3.2 ms/frame** |
| Processing speed | **~23 FPS** |
| Input resolution | 640 x 640 |
| Training epochs | 100 (patience=20) |
| Training batch size | 32 |

**Detected vehicle classes (8 total):**

| Class | Display Color |
|-------|--------------|
| car | Green |
| truck | Orange |
| bus | Red |
| motorbike | Magenta |
| van | Yellow |
| threewheel | Cyan |
| ambulance | Dark Red (triggers emergency routing) |
| bicycle | Purple |

---

## Congestion Levels

Congestion is calculated from the number of unique tracked vehicles (active ByteTrack IDs) visible in the current frame:

| Level | Threshold | Display Color |
|-------|-----------|---------------|
| LOW | 0 - 5 vehicles | Green |
| MEDIUM | 6 - 15 vehicles | Orange |
| HIGH | > 15 vehicles | Red |

When an ambulance class is detected in any frame, the payload sets `ambulance_detected: true`, which the Java Command Center uses to trigger a green wave emergency routing command.

---

## Telemetry Payload

Every second, the Python edge node serializes a JSON object and sends it over TCP with a newline delimiter (`\n`) so Java's `BufferedReader.readLine()` can detect message boundaries cleanly:

```json
{
  "timestamp": "2025-05-08 14:32:01",
  "vehicle_count": 12,
  "congestion_status": "MEDIUM",
  "class_counts": {
    "car": 7,
    "truck": 2,
    "bus": 1,
    "motorbike": 1,
    "van": 1,
    "threewheel": 0,
    "ambulance": 0,
    "bicycle": 0
  },
  "ambulance_detected": false,
  "fps": 23.1
}
```

---

## Python Edge Node

### How It Works

**1. Model loading**
`VehicleDetector` initializes an ONNX Runtime session with CUDA fallback to CPU. All graph optimizations are enabled. The model input tensor is `[1, 3, 640, 640]` (batch, channels, height, width) and output is `[1, 12, 8400]` (4 box coordinates + 8 class scores across 8400 anchor proposals).

**2. Preprocessing pipeline per frame**
Resize to 640x640 → BGR to RGB → normalize to [0,1] → transpose HWC to BCHW → cast to float32.

**3. Class confidence bias correction**
The training dataset contained highway footage with a heavy class imbalance toward bus and truck. A per-class weight vector is applied after inference to correct this before NMS:

```python
CLASS_CONFIDENCE_WEIGHTS = [
    1.30,  # car        (boost)
    0.75,  # truck      (suppress)
    0.70,  # bus        (suppress, biggest problem)
    1.20,  # motorbike  (boost)
    1.20,  # van        (boost)
    1.00,  # threewheel (neutral)
    1.00,  # ambulance  (neutral)
    1.10,  # bicycle    (slight boost)
]
```

**4. ByteTrack**
The `supervision` library's ByteTrack implementation assigns persistent unique integer IDs to each tracked vehicle across frames. `lost_track_buffer=30` keeps a track alive for 30 frames after it disappears (handles partial occlusion and brief exits from frame). The vehicle count shown in telemetry is the number of active tracker IDs in the current frame.

**5. Frame skipping**
Detection and tracking runs on every odd frame. Even frames reuse the previous annotated result for display only. This doubles effective display FPS without increasing inference cost.

**6. Non-blocking TCP send**
Each JSON payload is sent in a dedicated daemon thread so socket I/O never blocks the main video processing loop. If the connection drops, the broadcaster attempts one automatic reconnection on the next send cycle.

### Project Structure

```
python_edge_node/
├── edge_node.py        # Main pipeline: detection, tracking, TCP broadcasting, display
├── diagnose.py         # Pre-run diagnostic tool (checks deps, model, video, ONNX)
└── requirements.txt
```

### Prerequisites

- Python 3.10 or higher
- The trained ONNX model file (`best.onnx`) — see Model section below
- A traffic intersection video file (`traffic.mp4`)

### Setup

```bash
cd python_edge_node
pip install -r requirements.txt
```

Update the two path constants at the top of `edge_node.py`:

```python
MODEL_PATH = "path/to/your/best.onnx"
VIDEO_PATH = "path/to/your/traffic.mp4"
```

Run the diagnostic first to catch missing files or broken dependencies:

```bash
python diagnose.py
```

If all checks pass, start the edge node:

```bash
python edge_node.py
```

> Start the Java Command Center first. The Python node will keep retrying the TCP connection automatically if Java is not yet listening.

---

## Java Command Center

The Java application is a strictly multithreaded desktop GUI that acts as the intelligence layer above the Python edge node.

### Threading Architecture

| Thread | Responsibility |
|--------|---------------|
| JavaFX Application Thread | All GUI updates (vehicle count, congestion badge, class bars, chart) |
| Network Listener Thread | Continuous `ServerSocket.accept()` and `BufferedReader.readLine()` loop |
| JDBC Logger Thread | Background `PreparedStatement` writes to SQLite on every received payload |

The network and database threads never touch the GUI directly. All UI updates are dispatched via `Platform.runLater()` to prevent thread-safety violations.

### GUI Components

**Live Dashboard**
- Congestion badge (LOW / MEDIUM / HIGH) with color-coded background
- Total vehicle count and per-class breakdown bars
- Ambulance alert indicator for emergency routing

**JFreeChart Live Graph**
- Time-series line chart of vehicle count over time
- Scrolling window, updates every second as new telemetry arrives
- Separate series for total count and congestion level encoding

**Emergency Controls**
- Green wave routing button (activated automatically on ambulance detection)
- Manual override buttons for individual intersection signal control
- Command log panel showing issued commands with timestamps

### Database Schema (SQLite via JDBC)

```sql
CREATE TABLE traffic_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    vehicle_count    INTEGER NOT NULL,
    congestion_status TEXT   NOT NULL,
    car_count        INTEGER,
    truck_count      INTEGER,
    bus_count        INTEGER,
    motorbike_count  INTEGER,
    van_count        INTEGER,
    threewheel_count INTEGER,
    ambulance_count  INTEGER,
    bicycle_count    INTEGER,
    ambulance_alert  INTEGER,
    fps              REAL,
    logged_at        TEXT    DEFAULT (datetime('now'))
);
```

### Build and Run

```bash
cd java_command_center
mvn clean javafx:run
```

> Java Command Center must be running and listening before the Python edge node starts streaming.

---

## Model Training Details

Training was performed on Kaggle using a T4 GPU. The notebook combined two public vehicle detection datasets to reach 11,218 annotated images covering diverse traffic scenes including intersections, highways, and urban roads.

Key training configuration (from `training_args.yaml`):

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv11n (nano, pretrained) |
| Epochs | 100 (early stop patience=20) |
| Batch size | 32 |
| Image size | 640 x 640 |
| Optimizer | Auto |
| Augmentation | Mosaic, MixUp, RandAugment, HSV, flip |
| AMP (mixed precision) | Enabled |
| Device | Kaggle T4 GPU |

> The trained `best.onnx` model is not included in this repository due to file size. Re-train using the Kaggle notebook or contact the author for the model file.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| AI Training | Kaggle T4 GPU + Ultralytics YOLOv11 |
| AI Inference | ONNX Runtime (Python, CPU/CUDA) |
| Vehicle Tracking | ByteTrack via supervision library |
| Video Processing | OpenCV |
| Network Layer | Python `socket` (TCP client) / Java `ServerSocket` (TCP server) |
| GUI Framework | JavaFX |
| Data Visualization | JFreeChart |
| Database | SQLite via JDBC |
| Build Tool | Maven |

---

## Planned Improvements

- [ ] Replace video file source with live RTSP camera stream
- [ ] Add multi-intersection support (multiple edge nodes → one command center)
- [ ] Per-lane congestion tracking using grid-based region-of-interest zones
- [ ] Historical trend analysis tab in the Java GUI
- [ ] Export traffic logs to CSV from the dashboard

---

## Author

**Rao Hamza Bilal**

---

## License

This project is currently unlicensed. All rights reserved by the author.
