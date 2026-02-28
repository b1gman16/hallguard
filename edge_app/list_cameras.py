import cv2
import time

def try_open_camera(cam_id: int) -> bool:
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return False

    ok_reads = 0
    start = time.time()
    while time.time() - start < 2.0:  # try for 2 seconds
        ret, frame = cap.read()
        if ret and frame is not None:
            ok_reads += 1
            break

    cap.release()
    return ok_reads > 0

def main():
    print("=== Camera scan (0-5) ===")
    found = []
    for cam_id in range(0, 6):
        ok = try_open_camera(cam_id)
        print(f"Camera {cam_id}: {'OK' if ok else 'NO'}")
        if ok:
            found.append(cam_id)

    if found:
        print("[DONE] Working camera IDs:", found)
        print("Tip: One is your laptop cam, one should be DroidCam.")
    else:
        print("[ERROR] No cameras found. Check Windows camera permissions or DroidCam connection.")

if __name__ == "__main__":
    main()
