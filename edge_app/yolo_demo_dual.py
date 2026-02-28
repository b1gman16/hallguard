import cv2
from pathlib import Path

from camera import CameraStream
from roi import load_roi, draw_roi, roi_bbox
from yolo import YoloDetector

PERSON_CLASS_ID = 0  # COCO "person"

def draw_person_dets(frame, dets, offset_x=0, offset_y=0):
    for d in dets:
        if d["cls"] != PERSON_CLASS_ID:
            continue
        conf = d["conf"]
        x1, y1, x2, y2 = d["xyxy"]

        # shift boxes back to full-frame coordinates
        x1 += offset_x; x2 += offset_x
        y1 += offset_y; y2 += offset_y

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def safe_crop(frame, roi):
    x1, y1, x2, y2 = roi_bbox(roi)
    x1 = max(x1, 0); y1 = max(y1, 0)
    x2 = min(x2, frame.shape[1] - 1)
    y2 = min(y2, frame.shape[0] - 1)

    # avoid empty slices
    if x2 <= x1 or y2 <= y1:
        return None, (0, 0)

    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1)

def main():
    cam0_id = 0
    cam2_id = 2

    # Use 640x480 first for speed/stability
    cam0 = CameraStream(cam0_id, width=640, height=480)
    cam2 = CameraStream(cam2_id, width=640, height=480)

    ROOT = Path(__file__).resolve().parents[1]
    roi0 = load_roi(str(ROOT / "config" / "roi_cam0.json"))
    roi2 = load_roi(str(ROOT / "config" / "roi_cam2.json"))

    print("ROI0 loaded:", roi0 is not None, "points:", 0 if roi0 is None else len(roi0))
    print("ROI2 loaded:", roi2 is not None, "points:", 0 if roi2 is None else len(roi2))

    if roi0 is None or roi2 is None:
        print("[ERROR] Missing ROI(s). Run save_roi.py again.")
        return

    # One YOLO model instance is fine for both
    detector = YoloDetector(model_path="yolov8n.pt", conf=0.35)

    print("[INFO] Dual-camera YOLO demo running. Press Q to quit.")

    while True:
        f0 = cam0.read()
        f2 = cam2.read()
        if f0 is None or f2 is None:
            print("[ERROR] Failed to read from a camera.")
            break

        # Draw ROI outlines
        draw_roi(f0, roi0)
        draw_roi(f2, roi2)

        # --- Cam0 inference on ROI crop ---
        crop0, (ox0, oy0) = safe_crop(f0, roi0)
        if crop0 is not None:
            dets0 = detector.detect(crop0)
            draw_person_dets(f0, dets0, offset_x=ox0, offset_y=oy0)

        # --- Cam2 inference on ROI crop ---
        crop2, (ox2, oy2) = safe_crop(f2, roi2)
        if crop2 is not None:
            dets2 = detector.detect(crop2)
            draw_person_dets(f2, dets2, offset_x=ox2, offset_y=oy2)

        # Overlays
        cv2.putText(f0, f"Cam{cam0_id} FPS: {cam0.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f2, f"Cam{cam2_id} FPS: {cam2.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("HallGuard YOLO - Camera 0", f0)
        cv2.imshow("HallGuard YOLO - Camera 2 (DroidCam)", f2)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cam0.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
