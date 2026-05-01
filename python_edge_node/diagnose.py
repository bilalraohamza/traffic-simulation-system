
# diagnose.py
# Run this to identify exactly what is failing before running edge_node.py

import sys
import os

print("=" * 60)
print("TRAFFIC SYSTEM DIAGNOSTIC")
print("=" * 60)

# ── Step 1: Check Python version ──────────────────────────────────
print(f"\n[1] Python version: {sys.version}")

# ── Step 2: Check all dependencies ────────────────────────────────
print("\n[2] Checking dependencies...")
dependencies = {
    "onnxruntime" : "onnxruntime",
    "cv2"         : "opencv-python",
    "supervision" : "supervision",
    "numpy"       : "numpy",
}

all_ok = True
for module, package in dependencies.items():
    try:
        imported = __import__(module)
        version  = getattr(imported, '__version__', 'unknown')
        print(f"    {package:<25} OK  (version {version})")
    except ImportError:
        print(f"    {package:<25} MISSING - run: pip install {package}")
        all_ok = False

# ── Step 3: Check model file ───────────────────────────────────────
print("\n[3] Checking model file...")
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "model", "best.onnx")
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"    best.onnx found  ({size_mb:.1f} MB)")
else:
    print(f"    best.onnx MISSING at: {model_path}")
    print(f"    Check your folder structure.")
    all_ok = False

# ── Step 4: Check video file ───────────────────────────────────────
print("\n[4] Checking video file...")
video_path = os.path.join(script_dir, "traffic.mp4")
if os.path.exists(video_path):
    size_mb = os.path.getsize(video_path) / 1024 / 1024
    print(f"    traffic.mp4 found  ({size_mb:.1f} MB)")
else:
    print(f"    traffic.mp4 MISSING at: {video_path}")
    print(f"    Download it from Pixabay and place it at the path above.")
    all_ok = False

# ── Step 5: Test OpenCV can open the video ─────────────────────────
print("\n[5] Testing video open...")
try:
    import cv2
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                fps  = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"    Video opens OK")
                print(f"    Resolution : {w}x{h}")
                print(f"    FPS        : {fps:.1f}")
                print(f"    Frames     : {frames}")
            else:
                print(f"    Video opened but cannot read frames")
                print(f"    File may be corrupted. Re-download it.")
                all_ok = False
            cap.release()
        else:
            print(f"    OpenCV cannot open this video file")
            print(f"    Try re-downloading from Pixabay")
            all_ok = False
    else:
        print(f"    Skipped - video file missing")
except Exception as e:
    print(f"    Error: {e}")
    all_ok = False

# ── Step 6: Test ONNX model loads ─────────────────────────────────
print("\n[6] Testing ONNX model load...")
try:
    import onnxruntime as ort
    if os.path.exists(model_path):
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        inp  = session.get_inputs()[0]
        out  = session.get_outputs()[0]
        print(f"    ONNX model loads OK")
        print(f"    Input  : {inp.name} {inp.shape}")
        print(f"    Output : {out.name} {out.shape}")
        print(f"    Providers: {session.get_providers()}")
    else:
        print(f"    Skipped - model file missing")
except Exception as e:
    print(f"    Error loading ONNX model: {e}")
    all_ok = False

# ── Step 7: Test single inference ─────────────────────────────────
print("\n[7] Testing single inference run...")
try:
    import onnxruntime as ort
    import numpy as np
    if os.path.exists(model_path):
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        # Create a dummy input tensor (all zeros)
        dummy_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
        output = session.run(None, {"images": dummy_input})
        print(f"    Inference OK")
        print(f"    Output shape: {output[0].shape}")
    else:
        print(f"    Skipped - model file missing")
except Exception as e:
    print(f"    Inference error: {e}")
    all_ok = False

# ── Step 8: Test OpenCV window ────────────────────────────────────
print("\n[8] Testing OpenCV display window...")
try:
    import cv2
    import numpy as np
    test_frame = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Window - Press Q to close",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 1)
    cv2.imshow("Diagnostic Test Window", test_frame)
    print(f"    Window opened. Press Q to close it and continue.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print(f"    Window closed OK")
except Exception as e:
    print(f"    Window error: {e}")
    print(f"    This may be a display driver issue on Windows")
    all_ok = False

# ── Final summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED - edge_node.py should work correctly")
else:
    print("SOME CHECKS FAILED - fix the issues above before running edge_node.py")
print("=" * 60)