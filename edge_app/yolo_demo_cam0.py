import cv2
from pathlib import Path

from camera import CameraStream
from roi import load_roi, draw_roi, roi_bbox
from yolo import YoloDetector

# COCO class id for "person" in YOLOv8 pretrained models
PERSON_CLASS_ID = 0

def draw_dets(frame, dets, offset_x=0, offset_y=0):
    for d in dets:
        cls = d["cls"]
        conf = d["conf"]
        x1, y1, x2, y2 = d["xyxy"]

        # shift boxes back into full-frame coordinates
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y

        # Only draw persons for now (cleaner)
        if cls != PERSON_CLASS_ID:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def main():
    cam_id = 0

    cam = CameraStream(cam_id, width=640, height=480)

    ROOT = Path(__file__).resolve().parents[1]
    roi_path = ROOT / "config" / "roi_cam0.json"
    roi = load_roi(str(roi_path))

    print("ROI file:", roi_path, "exists:", roi_path.exists())
    print("ROI loaded:", roi is not None, "points:", 0 if roi is None else len(roi))

    if roi is None:
        print("[ERROR] ROI not found/invalid. Run save_roi.py first.")
        return

    det = YoloDetector(model_path="yolov8n.pt", conf=0.4)

    print("[INFO] YOLO demo running on ROI crop. Press Q to quit.")

    while True:
        frame = cam.read()
        if frame is None:
            print("[ERROR] Failed to read frame.")
            break

        # draw ROI polygon
        draw_roi(frame, roi)

        # crop to ROI bounding box (fast)
        x1, y1, x2, y2 = roi_bbox(roi)
        x1 = max(x1, 0); y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1] - 1)
        y2 = min(y2, frame.shape[0] - 1)

        crop = frame[y1:y2, x1:x2]

        # run YOLO on the crop
        dets = det.detect(crop)

        # draw detections back onto the full frame
        draw_dets(frame, dets, offset_x=x1, offset_y=y1)

        # FPS overlay (camera capture FPS, not YOLO FPS)
        cv2.putText(frame, f"Cam{cam_id} FPS: {cam.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("HallGuard YOLO Demo (Cam0)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
