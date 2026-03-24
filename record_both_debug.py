import cv2
import time
from pathlib import Path

print("Starting recorder...", flush=True)

cam0 = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(2)

print("cam0 opened:", cam0.isOpened(), flush=True)
print("cam1 opened:", cam1.isOpened(), flush=True)

if not cam0.isOpened():
    raise RuntimeError("Could not open camera 0")
if not cam1.isOpened():
    raise RuntimeError("Could not open camera 2")

time.sleep(2)

ret0, frame0 = cam0.read()
ret1, frame1 = cam1.read()

print("first read cam0:", ret0, flush=True)
print("first read cam1:", ret1, flush=True)

if not ret0 or frame0 is None:
    raise RuntimeError("Camera 0 did not return a valid frame")
if not ret1 or frame1 is None:
    raise RuntimeError("Camera 2 did not return a valid frame")

h0, w0 = frame0.shape[:2]
h1, w1 = frame1.shape[:2]

print(f"Camera 0 size: {w0}x{h0}", flush=True)
print(f"Camera 2 size: {w1}x{h1}", flush=True)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out0 = cv2.VideoWriter("cam0_test.avi", fourcc, 20.0, (w0, h0))
out1 = cv2.VideoWriter("cam2_test.avi", fourcc, 20.0, (w1, h1))

print("writer0 opened:", out0.isOpened(), flush=True)
print("writer1 opened:", out1.isOpened(), flush=True)

if not out0.isOpened():
    raise RuntimeError("Could not open VideoWriter for cam0_test.avi")
if not out1.isOpened():
    raise RuntimeError("Could not open VideoWriter for cam2_test.avi")

start = time.time()
duration = 10
count0 = 0
count1 = 0

while time.time() - start < duration:
    ret0, frame0 = cam0.read()
    ret1, frame1 = cam1.read()

    if ret0 and frame0 is not None:
        out0.write(frame0)
        count0 += 1

    if ret1 and frame1 is not None:
        out1.write(frame1)
        count1 += 1

cam0.release()
cam1.release()
out0.release()
out1.release()

print(f"Saved cam0_test.avi with {count0} frames", flush=True)
print(f"Saved cam2_test.avi with {count1} frames", flush=True)
print("Done.", flush=True)
