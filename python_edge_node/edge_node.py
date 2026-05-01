# =============================================================================
# edge_node.py
# Python Edge Node - Distributed Real-Time Traffic Simulation System
# Wuhan Institute of Technology - Advanced Programming Design Course
#
# This script performs the following operations in sequence:
#   1. Loads the trained YOLOv11n ONNX model for local AI inference
#   2. Opens a local traffic video file using OpenCV
#   3. Runs vehicle detection on every frame using ONNX Runtime
#   4. Applies ByteTrack algorithm to assign unique IDs to each vehicle
#   5. Calculates real-time congestion status from tracked vehicle count
#   6. Broadcasts JSON telemetry over TCP socket to Java Command Center
#   7. Displays the annotated video feed in a real-time window
# =============================================================================

import cv2                          # OpenCV for video reading and display
import numpy as np                  # NumPy for tensor operations
import onnxruntime as ort           # ONNX Runtime for local AI inference
import supervision as sv            # Supervision library for ByteTrack
import socket                       # Python socket for TCP broadcasting
import json                         # JSON for telemetry payload formatting
import time                         # Time for FPS calculation and intervals
import threading                    # Threading for non-blocking socket sends
from datetime import datetime       # Datetime for timestamp generation
from collections import defaultdict # For per-class vehicle counting

# =============================================================================
# CONFIGURATION
# All paths and parameters are defined here for easy modification
# =============================================================================

# Path to your trained ONNX model downloaded from Kaggle
MODEL_PATH = r"D:/Study/University/Semester 7/Advance Programming/traffic_system/model/best.onnx"

# Path to your traffic intersection video downloaded from Pixabay
VIDEO_PATH = r"D:/Study/University/Semester 7/Advance Programming/traffic_system/python_edge_node/traffic.mp4"

# Class names in the exact order used during training
# Order matches data.yaml: 0=car, 1=truck, 2=bus, 3=motorbike,
#                          4=van, 5=threewheel, 6=ambulance, 7=bicycle
CLASS_NAMES = [
    'car', 'truck', 'bus', 'motorbike',
    'van', 'threewheel', 'ambulance', 'bicycle'
]

# ONNX model input configuration
# These values must match exactly how the model was trained on Kaggle
INPUT_WIDTH    = 640    # Model input width in pixels
INPUT_HEIGHT   = 640    # Model input height in pixels
INPUT_NAME     = "images"    # ONNX input tensor name
OUTPUT_NAME    = "output0"   # ONNX output tensor name

# Detection confidence threshold
# Boxes with confidence below this value are discarded
CONFIDENCE_THRESHOLD = 0.40

# Non-Maximum Suppression IoU threshold
# Higher value = more overlapping boxes kept
NMS_IOU_THRESHOLD = 0.45

# Congestion thresholds based on unique tracked vehicle count
# These define when congestion status changes level
CONGESTION_LOW_MAX    = 5    # 0-5 vehicles = LOW
CONGESTION_MEDIUM_MAX = 15   # 6-15 vehicles = MEDIUM
                              # >15 vehicles = HIGH

# TCP socket configuration for communication with Java Command Center
TCP_HOST = "localhost"  # Java server runs on the same machine
TCP_PORT = 9999         # Java ServerSocket listens on this port

# How often to send telemetry to Java (in seconds)
TELEMETRY_INTERVAL = 1.0

# Color definitions for bounding box visualization (BGR format for OpenCV)
# Each class gets a distinct color for clear visual identification
CLASS_COLORS = {
    'car'       : (0,   255,  0  ),  # Green
    'truck'     : (255, 165,  0  ),  # Orange
    'bus'       : (0,   0,   255 ),  # Red
    'motorbike' : (255, 0,   255 ),  # Magenta
    'van'       : (0,   255, 255 ),  # Yellow
    'threewheel': (255, 255,  0  ),  # Cyan
    'ambulance' : (0,   0,   128 ),  # Dark Red (emergency vehicle)
    'bicycle'   : (128, 0,   128 ),  # Purple
}

# =============================================================================
# ONNX MODEL LOADER
# Loads the trained YOLOv11n model and prepares it for inference
# =============================================================================

class VehicleDetector:
    """
    Wraps the ONNX Runtime session for YOLOv11n vehicle detection.
    Handles model loading, preprocessing, inference, and postprocessing.
    """

    def __init__(self, model_path):
        """
        Initializes the ONNX Runtime inference session.
        Uses CPU execution provider for maximum compatibility on Windows 11.
        GPU provider would require CUDA toolkit which is optional.
        """
        print(f"[DETECTOR] Loading ONNX model from: {model_path}")

        # Create inference session with CPU execution provider
        # CPUExecutionProvider works on all Windows machines without CUDA
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Store input and output metadata for preprocessing
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"[DETECTOR] Model loaded successfully")
        print(f"[DETECTOR] Input  : {self.input_name} {self.input_shape}")
        print(f"[DETECTOR] Output : {self.output_name}")
        print(f"[DETECTOR] Classes: {CLASS_NAMES}")

    def preprocess(self, frame):
        """
        Converts an OpenCV BGR frame into the tensor format expected by ONNX.
        Steps:
          1. Resize frame to 640x640 (model input size)
          2. Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
          3. Normalize pixel values from 0-255 to 0.0-1.0
          4. Transpose from HWC to BCHW format (batch, channels, height, width)
          5. Cast to float32 (required by ONNX Runtime)
        """
        # Store original dimensions for scaling boxes back to display size
        self.orig_h, self.orig_w = frame.shape[:2]

        # Resize to model input dimensions
        resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

        # Convert color space BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1 range and convert to float32
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose HWC (640,640,3) to CHW (3,640,640)
        chw = normalized.transpose(2, 0, 1)

        # Add batch dimension: CHW -> BCHW (1,3,640,640)
        bchw = np.expand_dims(chw, axis=0)

        return bchw

    def postprocess(self, output, conf_threshold, iou_threshold):
        """
        Parses the raw ONNX output tensor into detection results.

        YOLOv11n output shape: (1, 12, 8400)
          - 8400 = number of anchor proposals
          - 12   = 4 box coordinates + 8 class confidence scores

        Box format from ONNX: [cx, cy, w, h] normalized to input size
        We convert to [x1, y1, x2, y2] in original image pixel coordinates.
        """
        # Remove batch dimension: (1, 12, 8400) -> (12, 8400)
        predictions = output[0][0]

        # Transpose to (8400, 12) for easier row-wise processing
        predictions = predictions.T

        # Split into box coordinates and class scores
        boxes_raw   = predictions[:, :4]   # (8400, 4) - cx, cy, w, h
        class_scores = predictions[:, 4:]  # (8400, 8) - one score per class

        # Get the class with highest confidence for each proposal
        class_ids   = np.argmax(class_scores, axis=1)    # (8400,)
        confidences = np.max(class_scores, axis=1)       # (8400,)

        # ── Class confidence bias correction ─────────────────────────────
        # Dataset 2 was heavily dominated by bus and truck classes from
        # highway footage, creating a classification bias toward these classes.
        # These multipliers reduce bus/truck dominance in ambiguous detections
        # and boost car/van/motorbike which were underrepresented.
        CLASS_CONFIDENCE_WEIGHTS = np.array([
            1.30,   # car        - significantly boost car confidence
            0.75,   # truck      - reduce truck over-detection
            0.70,   # bus        - reduce bus over-detection (biggest problem)
            1.20,   # motorbike  - boost motorbike
            1.20,   # van        - boost van
            1.00,   # threewheel - neutral
            1.00,   # ambulance  - neutral
            1.10,   # bicycle    - slight boost
        ])

        # Apply weights element-wise to all class score columns
        # Shape: (N, 8) * (8,) = (N, 8) via numpy broadcasting
        class_scores = class_scores * CLASS_CONFIDENCE_WEIGHTS

        # Recalculate class IDs and confidences after applying weights
        class_ids   = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        # Filter out low-confidence detections
        mask = confidences > conf_threshold
        boxes_raw    = boxes_raw[mask]
        confidences  = confidences[mask]
        class_ids    = class_ids[mask]

        if len(boxes_raw) == 0:
            # No detections above threshold
            return np.array([]), np.array([]), np.array([])

        # Scale factor to convert from model input size to original frame size
        scale_x = self.orig_w / INPUT_WIDTH
        scale_y = self.orig_h / INPUT_HEIGHT

        # Convert cx,cy,w,h to x1,y1,x2,y2 in original pixel coordinates
        cx = boxes_raw[:, 0] * scale_x
        cy = boxes_raw[:, 1] * scale_y
        w  = boxes_raw[:, 2] * scale_x
        h  = boxes_raw[:, 3] * scale_y

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Stack into (N, 4) array of pixel-coordinate boxes
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Apply Non-Maximum Suppression to remove duplicate detections
        # cv2.dnn.NMSBoxes expects boxes as list of [x, y, w, h]
        boxes_xywh = [[
            float(x1[i]), float(y1[i]),
            float(x2[i] - x1[i]), float(y2[i] - y1[i])
        ] for i in range(len(boxes_xyxy))]

        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences.tolist(),
            conf_threshold,
            iou_threshold
        )

        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])

        # Flatten indices (OpenCV returns nested array)
        indices = indices.flatten()

        return (
            boxes_xyxy[indices],    # Final bounding boxes (N, 4)
            confidences[indices],   # Confidence scores (N,)
            class_ids[indices]      # Class IDs (N,)
        )

    def detect(self, frame):
        """
        Runs the full detection pipeline on a single frame.
        Returns boxes, confidences, and class IDs for all detected vehicles.
        """
        # Preprocess frame into model input tensor
        input_tensor = self.preprocess(frame)

        # Run ONNX inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        # Parse raw output into detection results
        boxes, confidences, class_ids = self.postprocess(
            outputs, CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD
        )

        return boxes, confidences, class_ids


# =============================================================================
# CONGESTION CALCULATOR
# Determines traffic congestion level from unique vehicle count
# =============================================================================

def calculate_congestion(vehicle_count):
    """
    Converts a raw vehicle count into a human-readable congestion status.
    Thresholds are defined in the configuration section above.

    Returns:
        str: "LOW", "MEDIUM", or "HIGH"
    """
    if vehicle_count <= CONGESTION_LOW_MAX:
        return "LOW"
    elif vehicle_count <= CONGESTION_MEDIUM_MAX:
        return "MEDIUM"
    else:
        return "HIGH"


# =============================================================================
# TCP SOCKET BROADCASTER
# Manages the connection to the Java Command Center and sends JSON payloads
# =============================================================================

class TelemetryBroadcaster:
    """
    Handles TCP socket connection to the Java Command Center.
    Sends JSON telemetry payloads containing traffic metrics.
    Runs socket operations in a separate thread to avoid blocking video loop.
    """

    def __init__(self, host, port):
        """
        Initializes broadcaster with connection parameters.
        Does not connect immediately - connection happens on first send.
        """
        self.host       = host
        self.port       = port
        self.socket     = None
        self.connected  = False
        self.lock       = threading.Lock()  # Thread safety for socket access

    def connect(self):
        """
        Attempts to establish TCP connection to Java Command Center.
        Java must be started first and listening on the port.
        Returns True if connection succeeded, False otherwise.
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)            # 5 second connection timeout
            self.socket.connect((self.host, self.port))
            self.socket.settimeout(None)         # Remove timeout after connect
            self.connected = True
            print(f"[SOCKET] Connected to Java Command Center at "
                  f"{self.host}:{self.port}")
            return True
        except ConnectionRefusedError:
            print(f"[SOCKET] Connection refused. Is Java Command Center "
                  f"running on port {self.port}?")
            self.connected = False
            return False
        except Exception as e:
            print(f"[SOCKET] Connection error: {e}")
            self.connected = False
            return False

    def send_telemetry(self, payload_dict):
        """
        Serializes the payload dictionary to JSON and sends over TCP.
        Appends newline character as message delimiter for Java's
        BufferedReader.readLine() to detect message boundaries.

        If connection is lost, attempts one reconnection automatically.
        """
        with self.lock:
            if not self.connected:
                # Attempt reconnection silently
                self.connect()
                if not self.connected:
                    return  # Skip this payload if still not connected

            try:
                # Serialize to JSON string and add newline delimiter
                json_str  = json.dumps(payload_dict) + "\n"
                # Encode to bytes and send over socket
                self.socket.sendall(json_str.encode('utf-8'))

            except (BrokenPipeError, ConnectionResetError, OSError):
                # Connection was lost, mark as disconnected
                print("[SOCKET] Connection lost. Will retry on next send.")
                self.connected = False
                try:
                    self.socket.close()
                except:
                    pass

    def close(self):
        """Cleanly closes the TCP socket connection."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print("[SOCKET] Connection closed.")


# =============================================================================
# ANNOTATION RENDERER
# Draws detection results and tracking IDs on video frames
# =============================================================================

def draw_annotations(frame, boxes, class_ids, track_ids,
                      vehicle_count, congestion_status, fps):
    """
    Draws bounding boxes, class labels, tracking IDs, and system status
    overlay onto the video frame for real-time visualization.
    Uses larger text and higher contrast for better readability on
    high resolution traffic footage.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Scale font size based on frame resolution
    # Larger videos need larger text to remain readable
    scale_factor = w / 1280.0
    font_scale_label   = max(0.6,  0.7  * scale_factor)
    font_scale_overlay = max(0.8,  0.9  * scale_factor)
    font_thickness     = max(1, int(2 * scale_factor))
    box_thickness      = max(2, int(3 * scale_factor))

    # Draw each detected and tracked vehicle
    for i, (box, cls_id) in enumerate(zip(boxes, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        class_name = CLASS_NAMES[int(cls_id)]
        color      = CLASS_COLORS.get(class_name, (0, 255, 0))
        track_id   = int(track_ids[i]) if i < len(track_ids) else -1

        # Draw bounding box with thick border for visibility
        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                      color, box_thickness)

        # Build label: class name and unique tracking ID
        label = f"{class_name} #{track_id}"

        # Measure label size to draw background rectangle
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_label, font_thickness
        )

        # Position label above bounding box
        label_y = max(y1 - 5, label_h + 10)

        # Draw solid background rectangle for label readability
        cv2.rectangle(
            annotated,
            (x1, label_y - label_h - baseline - 4),
            (x1 + label_w + 6, label_y + baseline - 2),
            color, -1
        )

        # Draw white text over colored background
        cv2.putText(
            annotated, label,
            (x1 + 3, label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_label,
            (255, 255, 255),   # White text for maximum contrast
            font_thickness,
            cv2.LINE_AA
        )

    # ── Draw status overlay panel ─────────────────────────────────
    # Scale overlay size with frame resolution
    panel_w = int(380 * scale_factor)
    panel_h = int(180 * scale_factor)
    line_h  = int(panel_h / 5)

    # Draw semi-transparent dark background for overlay panel
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Congestion color indicator
    congestion_colors = {
        "LOW"   : (0,   220,  0  ),   # Bright green
        "MEDIUM": (0,   165, 255),    # Orange
        "HIGH"  : (0,   0,   220),    # Red
    }
    cong_color = congestion_colors.get(congestion_status, (255, 255, 255))

    padding = int(12 * scale_factor)

    # Line 1: System title
    cv2.putText(annotated, "TRAFFIC COMMAND SYSTEM",
                (padding, line_h),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_overlay * 0.85,
                (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Line 2: Vehicle count
    cv2.putText(annotated, f"Vehicles   : {vehicle_count}",
                (padding, line_h * 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_overlay,
                (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Line 3: Congestion status with matching color
    cv2.putText(annotated, f"Congestion : {congestion_status}",
                (padding, line_h * 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_overlay,
                cong_color, font_thickness + 1, cv2.LINE_AA)

    # Line 4: FPS counter
    cv2.putText(annotated, f"FPS        : {fps:.1f}",
                (padding, line_h * 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_overlay * 0.85,
                (200, 200, 200), font_thickness, cv2.LINE_AA)

    # Line 5: TCP status
    cv2.putText(annotated, "TCP : BROADCASTING TO JAVA",
                (padding, line_h * 5 - padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_overlay * 0.7,
                (0, 255, 180), font_thickness, cv2.LINE_AA)

    return annotated


# =============================================================================
# MAIN PROCESSING LOOP
# Orchestrates all components: detection, tracking, broadcasting, display
# =============================================================================

def main():
    """
    Main entry point for the Python Edge Node.
    Initializes all components and runs the real-time processing loop.
    """
    print("=" * 60)
    print("  PYTHON EDGE NODE - TRAFFIC SIMULATION SYSTEM")
    print("=" * 60)

    # ── Load ONNX model ───────────────────────────────────────────
    detector = VehicleDetector(MODEL_PATH)

    # ── Open video file ───────────────────────────────────────────
    print(f"\n[VIDEO] Opening: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {VIDEO_PATH}")
        print("        Check that traffic.mp4 exists at the path above.")
        return

    # Read video properties for display information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    video_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[VIDEO] Resolution : {video_w}x{video_h}")
    print(f"[VIDEO] FPS        : {video_fps:.1f}")
    print(f"[VIDEO] Frames     : {total_frames}")

    # ── Initialize ByteTrack via supervision library ──────────────
    # ByteTrack assigns stable unique IDs to vehicles across frames
    # lost_track_buffer = how many frames to keep a lost track alive
    tracker = sv.ByteTrack(
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
    )
    print("[TRACKER] ByteTrack initialized")

    # ── Initialize TCP broadcaster ────────────────────────────────
    broadcaster = TelemetryBroadcaster(TCP_HOST, TCP_PORT)
    print(f"[SOCKET] Attempting connection to Java on port {TCP_PORT}...")
    print("[SOCKET] If Java is not running yet, broadcaster will retry.")
    broadcaster.connect()

    # ── State variables for telemetry timing ─────────────────────
    last_telemetry_time = time.time()   # Tracks when we last sent JSON
    frame_count         = 0             # Total frames processed
    fps_start_time      = time.time()   # For FPS calculation
    current_fps         = 0.0           # Displayed FPS value

    print("\n[SYSTEM] Starting video processing loop...")
    print("[SYSTEM] Press 'Q' in the video window to quit.\n")

    # ── Main frame processing loop ────────────────────────────────
    while True:

        # Read next frame from video file
        ret, frame = cap.read()

        if not ret:
            print("[VIDEO] End of video reached. Restarting from beginning...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = sv.ByteTrack(lost_track_buffer=30)
            frame_count = 0
            continue

        frame_count += 1

        # Skip every other frame to maintain smooth FPS
        # Detection still runs at half frame rate which is sufficient
        # for accurate vehicle counting and tracking
        if frame_count % 2 == 0:
            # Show previous annotated frame without re-running inference
            if 'annotated_frame' in locals():
                display_frame = cv2.resize(annotated_frame, (1280, 720))
                cv2.namedWindow(
                    "Traffic Edge Node - YOLOv11n + ByteTrack",
                    cv2.WINDOW_NORMAL
                )
                cv2.resizeWindow(
                    "Traffic Edge Node - YOLOv11n + ByteTrack",
                    1280, 720
                )
                cv2.imshow(
                    "Traffic Edge Node - YOLOv11n + ByteTrack",
                    display_frame
                )
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # ── Run vehicle detection ──────────────────────────────────
        boxes, confidences, class_ids = detector.detect(frame)

        # ── Apply ByteTrack for vehicle tracking ───────────────────
        # ByteTrack requires detections in supervision Detections format
        if len(boxes) > 0:
            # Create supervision Detections object from our raw results
            detections = sv.Detections(
                xyxy       = boxes,
                confidence = confidences,
                class_id   = class_ids.astype(int),
            )
            # Update tracker with current frame detections
            # Returns detections with stable tracker_id assigned
            tracked_detections = tracker.update_with_detections(detections)

            # Extract tracking IDs, boxes, and class IDs from tracked results
            track_ids        = tracked_detections.tracker_id
            tracked_boxes    = tracked_detections.xyxy
            tracked_class_ids = tracked_detections.class_id

        else:
            # No detections this frame
            track_ids         = np.array([])
            tracked_boxes     = np.array([]).reshape(0, 4)
            tracked_class_ids = np.array([])

        # ── Calculate traffic metrics ──────────────────────────────
        # Unique vehicle count = number of active tracks this frame
        vehicle_count      = len(track_ids)
        congestion_status  = calculate_congestion(vehicle_count)

        # Count vehicles per class for detailed telemetry
        class_counts = defaultdict(int)
        for cls_id in tracked_class_ids:
            class_name = CLASS_NAMES[int(cls_id)]
            class_counts[class_name] += 1

        # Check if ambulance is present (triggers green wave in Java)
        ambulance_detected = class_counts.get('ambulance', 0) > 0

        # ── Calculate FPS every 30 frames ─────────────────────────
        if frame_count % 30 == 0:
            elapsed      = time.time() - fps_start_time
            current_fps  = 30.0 / elapsed if elapsed > 0 else 0
            fps_start_time = time.time()

        # ── Send telemetry to Java every TELEMETRY_INTERVAL seconds ─
        current_time = time.time()
        if current_time - last_telemetry_time >= TELEMETRY_INTERVAL:

            # Build the JSON payload that Java will parse
            payload = {
                "timestamp"         : datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                      ),
                "vehicle_count"     : int(vehicle_count),
                "congestion_status" : congestion_status,
                "class_counts"      : {
                    "car"       : int(class_counts.get('car', 0)),
                    "truck"     : int(class_counts.get('truck', 0)),
                    "bus"       : int(class_counts.get('bus', 0)),
                    "motorbike" : int(class_counts.get('motorbike', 0)),
                    "van"       : int(class_counts.get('van', 0)),
                    "threewheel": int(class_counts.get('threewheel', 0)),
                    "ambulance" : int(class_counts.get('ambulance', 0)),
                    "bicycle"   : int(class_counts.get('bicycle', 0)),
                },
                "ambulance_detected": ambulance_detected,
                "fps"               : round(current_fps, 1),
            }

            # Send payload in background thread to avoid blocking video loop
            send_thread = threading.Thread(
                target=broadcaster.send_telemetry,
                args=(payload,),
                daemon=True
            )
            send_thread.start()

            last_telemetry_time = current_time

            # Print status to console every telemetry interval
            print(f"[{payload['timestamp']}] "
                  f"Vehicles={vehicle_count:2d} | "
                  f"Congestion={congestion_status:<6} | "
                  f"FPS={current_fps:.1f} | "
                  f"Ambulance={'YES' if ambulance_detected else 'no'}")

        # ── Draw annotations on frame ──────────────────────────────
        annotated_frame = draw_annotations(
            frame,
            tracked_boxes,
            tracked_class_ids,
            track_ids,
            vehicle_count,
            congestion_status,
            current_fps
        )

        # ── Display annotated frame ────────────────────────────────
        # Resize display frame to fixed window size for clean presentation
        # ── Display annotated frame ──────────────────────────────────────
        # Resize to 1280x720 for clean HD display on Windows 11
        display_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.namedWindow(
            "Traffic Edge Node - YOLOv11n + ByteTrack",
            cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(
            "Traffic Edge Node - YOLOv11n + ByteTrack",
            1280, 720
        )
        cv2.imshow(
            "Traffic Edge Node - YOLOv11n + ByteTrack",
            display_frame
        )

        # Check for 'Q' key press to quit
        # cv2.waitKey(1) is required for OpenCV window to refresh
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[SYSTEM] User pressed Q. Shutting down...")
            break
        try:
            window_visible = cv2.getWindowProperty(
                "Traffic Edge Node - YOLOv11n + ByteTrack",
                cv2.WND_PROP_VISIBLE
            )
            if window_visible < 1:
                print("\n[SYSTEM] Window closed by user. Shutting down...")
                break
        except:
            break

    # ── Cleanup on exit ────────────────────────────────────────────
    cap.release()                   # Release video file handle
    cv2.destroyAllWindows()         # Close OpenCV display window
    broadcaster.close()             # Close TCP socket connection
    print("[SYSTEM] Edge Node stopped cleanly.")


# =============================================================================
# ENTRY POINT
# Script runs main() when executed directly
# =============================================================================

if __name__ == "__main__":
    main()