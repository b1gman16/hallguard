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

fps = 20.0
frame_interval = 1.0 / fps

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out0 = cv2.VideoWriter(str(out_dir / "cam0_test.avi"), fourcc, fps, (w0, h0))
out1 = cv2.VideoWriter(str(out_dir / "cam1_test.avi"), fourcc, fps, (w1, h1))

if not out0.isOpened():
    raise RuntimeError("Could not open writer for cam0_test.avi")
if not out1.isOpened():
    raise RuntimeError("Could not open writer for cam1_test.avi")

duration = 10  # seconds
start = time.time()
next_frame_time = start

frames_written = 0

while time.time() - start < duration:
    now = time.time()

    if now >= next_frame_time:
        frame0 = cam0.read()
        frame1 = cam1.read()

        if frame0 is not None:
            out0.write(frame0)

        if frame1 is not None:
            out1.write(frame1)

        frames_written += 1
        next_frame_time += frame_interval
    else:
        time.sleep(0.001)

cam0.release()
cam1.release()
out0.release()
out1.release()

actual_duration = time.time() - start
print(f"Saved cam0_test.avi and cam1_test.avi")
print(f"Frames written: {frames_written}")
print(f"Actual duration: {actual_duration:.2f} seconds")
print(f"Effective write FPS: {frames_written / actual_duration:.2f}")