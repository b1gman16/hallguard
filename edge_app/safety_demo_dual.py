import cv2
from pathlib import Path

from camera import CameraStream
from roi import load_roi, draw_roi, roi_bbox
from yolo import YoloDetector
from decision import SafetyDecision

PERSON_CLASS_ID = 0  # COCO "person"

def has_person(dets):
    return any(d["cls"] == PERSON_CLASS_ID for d in dets)

def safe_crop(frame, roi):
    x1, y1, x2, y2 = roi_bbox(roi)
    x1 = max(x1, 0); y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1] - 1)
    y2 = min(y2, frame.shape[0] - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def draw_status(frame, status, x=10, y=95):
    # Green for SAFE, Red for UNSAFE
    color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)
    cv2.putText(frame, f"STATUS: {status}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def main():
    cam0_id = 0
    cam2_id = 2

    cam0 = CameraStream(cam0_id, width=640, height=480)
    cam2 = CameraStream(cam2_id, width=640, height=480)

    ROOT = Path(__file__).resolve().parents[1]
    roi0 = load_roi(str(ROOT / "config" / "roi_cam0.json"))
    roi2 = load_roi(str(ROOT / "config" / "roi_cam2.json"))

    if roi0 is None or roi2 is None:
        print("[ERROR] Missing ROI(s). Run save_roi.py.")
        return

    detector = YoloDetector(model_path="yolov8n.pt", conf=0.35)

    # smoothing per camera
    dec0 = SafetyDecision(unsafe_on_count=3, safe_on_count=5)
    dec2 = SafetyDecision(unsafe_on_count=3, safe_on_count=5)

    print("[INFO] SAFE/UNSAFE demo running. Press Q to quit.")

    while True:
        f0 = cam0.read()
        f2 = cam2.read()
        if f0 is None or f2 is None:
            print("[ERROR] Failed to read from a camera.")
            break

        draw_roi(f0, roi0)
        draw_roi(f2, roi2)

        crop0 = safe_crop(f0, roi0)
        crop2 = safe_crop(f2, roi2)

        dets0 = detector.detect(crop0) if crop0 is not None else []
        dets2 = detector.detect(crop2) if crop2 is not None else []

        # "unsafe seen" == person seen (placeholder logic for now)
        unsafe0_seen = has_person(dets0)
        unsafe2_seen = has_person(dets2)

        status0 = dec0.update(unsafe0_seen)
        status2 = dec2.update(unsafe2_seen)

        # overlays
        cv2.putText(f0, f"Cam{cam0_id} FPS: {cam0.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f2, f"Cam{cam2_id} FPS: {cam2.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(f0, f"person_in_roi: {unsafe0_seen}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(f2, f"person_in_roi: {unsafe2_seen}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        draw_status(f0, status0)
        draw_status(f2, status2)

        cv2.imshow("HallGuard SAFE/UNSAFE - Camera 0", f0)
        cv2.imshow("HallGuard SAFE/UNSAFE - Camera 2 (DroidCam)", f2)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cam0.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
