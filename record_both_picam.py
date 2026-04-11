from pathlib import Path
import argparse
import cv2
import time
from edge_app.camera import CameraStream


def parse_args():
    parser = argparse.ArgumentParser(description="Record from both Pi cameras.")
    parser.add_argument("--minutes", type=float, default=None, help="Recording duration in minutes")
    parser.add_argument("--seconds", type=float, default=None, help="Recording duration in seconds")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview window")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--fps", type=float, default=20.0, help="Target recording FPS")
    return parser.parse_args()


args = parse_args()

if args.minutes is not None and args.seconds is not None:
    raise ValueError("Use either --minutes or --seconds, not both.")

if args.minutes is not None:
    duration = args.minutes * 60
elif args.seconds is not None:
    duration = args.seconds
else:
    duration = 10 * 60  # default: 10 minutes

show_preview = not args.no_preview
fps = args.fps
frame_interval = 1.0 / fps

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
file0 = out_dir / f"cam0_{timestamp}.avi"
file1 = out_dir / f"cam1_{timestamp}.avi"

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

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out0 = cv2.VideoWriter(str(file0), fourcc, fps, (w0, h0))
out1 = cv2.VideoWriter(str(file1), fourcc, fps, (w1, h1))

if not out0.isOpened():
    raise RuntimeError(f"Could not open writer for {file0}")
if not out1.isOpened():
    raise RuntimeError(f"Could not open writer for {file1}")

frames_written = 0
missed0 = 0
missed1 = 0

start = time.monotonic()
next_frame_time = start

try:
    while time.monotonic() - start < duration:
        now = time.monotonic()

        if now >= next_frame_time:
            frame0 = cam0.read()
            frame1 = cam1.read()

            if frame0 is not None:
                out0.write(frame0)
            else:
                missed0 += 1

            if frame1 is not None:
                out1.write(frame1)
            else:
                missed1 += 1

            if show_preview and frame0 is not None and frame1 is not None:
                if frame0.shape[0] != frame1.shape[0]:
                    target_h = min(frame0.shape[0], frame1.shape[0])
                    scale0 = target_h / frame0.shape[0]
                    scale1 = target_h / frame1.shape[0]
                    f0_disp = cv2.resize(frame0, (int(frame0.shape[1] * scale0), target_h))
                    f1_disp = cv2.resize(frame1, (int(frame1.shape[1] * scale1), target_h))
                else:
                    f0_disp = frame0
                    f1_disp = frame1

                combined = cv2.hconcat([f0_disp, f1_disp])
                cv2.imshow("Recording Preview", combined)

            frames_written += 1
            next_frame_time += frame_interval
        else:
            time.sleep(0.001)

        if show_preview:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

finally:
    cam0.release()
    cam1.release()
    out0.release()
    out1.release()
    cv2.destroyAllWindows()

actual_duration = time.monotonic() - start
print(f"Saved {file0} and {file1}")
print(f"Frames written: {frames_written}")
print(f"Missed frames: cam0={missed0}, cam1={missed1}")
print(f"Actual duration: {actual_duration:.2f} seconds")
print(f"Effective write FPS: {frames_written / actual_duration:.2f}")