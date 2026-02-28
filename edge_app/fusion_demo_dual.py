import cv2
from pathlib import Path
import time

from camera import CameraStream
from roi import load_roi, draw_roi, roi_bbox
from yolo import YoloDetector
from decision import SafetyDecision
from fusion import FusionEngine
from alarm import Alarm
from firebase_client import FirebaseLogger


PERSON_CLASS_ID = 0

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

def draw_status(frame, status, y=95):
    color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)
    cv2.putText(frame, f"STATUS: {status}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_fusion_overlay(frame, fusion_state, y=130):
    if fusion_state is None:
        cv2.putText(frame, "FUSION: none", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return

    txt = f"EVENT {fusion_state['event_id']} cams={fusion_state['cameras_seen']} dual={fusion_state['confirmed_dual']} handoff={fusion_state['handoff']}"
    cv2.putText(frame, txt, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

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
    dec0 = SafetyDecision(unsafe_on_count=3, safe_on_count=5)
    dec2 = SafetyDecision(unsafe_on_count=3, safe_on_count=5)

    fusion = FusionEngine(
        confirm_window_s=0.7,
        handoff_window_s=0.7,
        end_after_s=2.0,
        cooldown_s=5.0
    )

    alarm = Alarm(cooldown_s=5.0)

    firebase = FirebaseLogger(
        service_account_path="config/firebase_service_account.json",
        collection="events"
    )

    location_id = "bedroom-test"  # change later to your real location

    last_print = 0.0
    print("[INFO] Fusion demo running. Press Q to quit.")

    while True:
        f0 = cam0.read()
        f2 = cam2.read()
        if f0 is None or f2 is None:
            break

        draw_roi(f0, roi0)
        draw_roi(f2, roi2)

        dets0 = detector.detect(safe_crop(f0, roi0)) if roi0 else []
        dets2 = detector.detect(safe_crop(f2, roi2)) if roi2 else []

        status0 = dec0.update(has_person(dets0))
        status2 = dec2.update(has_person(dets2))

        fused_event, fused_status = fusion.update(status0, status2)

        if fused_status in ("started", "ended") and fused_event is not None:
            payload = {
                "event_id": fused_event["event_id"],
                "status": fused_status,
                "location_id": location_id,
                "cameras_seen": fused_event.get("cameras_seen", []),
                "confirmed_dual": fused_event.get("confirmed_dual", False),
                "handoff": fused_event.get("handoff", False),
                "start_time": fused_event.get("start_time", None),
                "last_update": fused_event.get("last_update", None),
                "client_time": time.time(),
            }

            if fused_status == "ended":
                payload["end_time"] = time.time()

            # Use doc_id = event_id so "ended" merges into same doc
            doc_id = f"event_{payload['event_id']}"
            firebase.log_event(payload, doc_id=doc_id)

            print("[FIREBASE] logged:", doc_id, fused_status)


        if fused_status == "started":
            alarm.trigger()

        # occasional terminal print (not every frame)
        now = time.time()
        if fused_status != "none" and now - last_print > 0.3:
            print("Fusion:", fused_status, fused_event)
            last_print = now

        # overlays
        cv2.putText(f0, f"Cam{cam0_id} FPS: {cam0.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(f2, f"Cam{cam2_id} FPS: {cam2.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        draw_status(f0, status0)
        draw_status(f2, status2)

        draw_fusion_overlay(f0, fused_event)
        draw_fusion_overlay(f2, fused_event)

        cv2.imshow("HallGuard Fusion - Camera 0", f0)
        cv2.imshow("HallGuard Fusion - Camera 2 (DroidCam)", f2)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cam0.release()
    cam2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
