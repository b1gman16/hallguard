import cv2
import json
import os
from camera import CameraStream
import numpy as np


points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point added: {x}, {y}")

def select_and_save(camera_id, filename):
    global points
    points = []

    cam = CameraStream(camera_id, width=640, height=480)

# Warm up: read a few frames to let the camera stabilize
    frame = None
    for _ in range(30):
        frame = cam.read()
        if frame is not None:
            pass


    if frame is None:
        print("[ERROR] Could not read frame")
        cam.release()
        return


    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    print("Click points around the railing.")
    print("Press S = save | R = reset | Q = cancel")

    while True:
        temp = frame.copy()

        for p in points:
            cv2.circle(temp, tuple(p), 5, (0, 255, 0), -1)

        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp, [pts], True, (0, 255, 0), 2)


        cv2.imshow("Select ROI", temp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and len(points) >= 3:
            break
        elif key == ord("r"):
            points = []
            print("ROI reset")
        elif key == ord("q"):
            points = []
            cam.release()
            cv2.destroyAllWindows()
            return

    cam.release()
    cv2.destroyAllWindows()

    os.makedirs("config", exist_ok=True)
    with open(f"config/{filename}", "w") as f:
        json.dump(points, f, indent=2)

    print(f"[OK] ROI saved to config/{filename}")

def main():
    select_and_save(0, "roi_cam0.json")
    select_and_save(2, "roi_cam2.json")

if __name__ == "__main__":
    main()
