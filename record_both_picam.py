from pathlib import Path
import cv2
import time
from edge_app.camera import CameraStream

out_dir = Path(".")

cam0 = CameraStream(0, width=640, height=480)
cam1 = CameraStream(1, width=640, height=480)

frame0 = cam0.read()
frame1 = cam1.read()

if frame0 is None:
    raise RuntimeError("Camera 0 did not return a frame")
if frame1 is None:
    raise RuntimeError("Camera 1 did not return a frame")

h0, w0 = frame0.shape[:2]
h1, w1 = frame1.shape[:2]

# Use 30 fps as a better default for Pi camera preview streams
fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out0 = cv2.VideoWriter(str(out_dir / "cam0_test.avi"), fourcc, fps, (w0, h0))
out1 = cv2.VideoWriter(str(out_dir / "cam1_test.avi"), fourcc, fps, (w1, h1))

if not out0.isOpened():
    raise RuntimeError("Could not open writer for cam0_test.avi")
if not out1.isOpened():
    raise RuntimeError("Could not open writer for cam1_test.avi")

duration = 10  # seconds
start = time.time()

while time.time() - start < duration:
    frame0 = cam0.read()
    frame1 = cam1.read()

    if frame0 is not None:
        out0.write(frame0)

    if frame1 is not None:
        out1.write(frame1)

cam0.release()
cam1.release()
out0.release()
out1.release()

print("Saved cam0_test.avi and cam1_test.avi")