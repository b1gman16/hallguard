import json
import time
from pathlib import Path

import cv2

from camera import CameraStream
from roi import load_roi, draw_roi, roi_bbox
from yolo import YoloDetector
from decision import SafetyDecision
from fusion import FusionEngine
from alarm import Alarm


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def safe_crop(frame, roi):
    """Crop frame to ROI bounding box. Returns (crop, offset_x, offset_y)."""
    x1, y1, x2, y2 = roi_bbox(roi)
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), frame.shape[1] - 1)
    y2 = min(int(y2), frame.shape[0] - 1)

    if x2 <= x1 or y2 <= y1:
        return None, 0, 0

    crop = frame[y1:y2, x1:x2]
    return crop, x1, y1


def has_person(dets, person_class_id=0) -> bool:
    return any(d.get("cls") == person_class_id for d in dets)


def draw_status(frame, status: str, y=95):
    color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)
    cv2.putText(
        frame,
        f"STATUS: {status}",
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )


def draw_fusion_overlay(frame, fusion_event, y=130):
    if fusion_event is None:
        cv2.putText(
            frame,
            "FUSION: none",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return

    txt = (
        f"EVENT {fusion_event.get('event_id')} "
        f"cams={fusion_event.get('cameras_seen')} "
        f"dual={fusion_event.get('confirmed_dual')} "
        f"handoff={fusion_event.get('handoff')}"
    )
    cv2.putText(
        frame,
        txt,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def main():
    ROOT = Path(__file__).resolve().parents[1]
    cfg = load_config(str(ROOT / "config" / "system_config.json"))

    location_id = cfg.get("location_id", "unknown-location")

    cam_cfg = cfg["cameras"]
    cam0_id = cam_cfg["cam0_id"]
    cam2_id = cam_cfg["cam2_id"]
    width = cam_cfg.get("width", 640)
    height = cam_cfg.get("height", 480)

    roi_cfg = cfg["roi"]
    roi0_path = ROOT / roi_cfg["cam0_path"]
    roi2_path = ROOT / roi_cfg["cam2_path"]

    yolo_cfg = cfg["yolo"]
    model_path = (
        str(ROOT / yolo_cfg.get("model_path", "yolov8n.pt"))
        if not Path(yolo_cfg.get("model_path", "")).is_absolute()
        else yolo_cfg["model_path"]
    )
    conf = float(yolo_cfg.get("conf", 0.35))
    person_class_id = int(yolo_cfg.get("person_class_id", 0))
    imgsz = int(yolo_cfg.get("imgsz", 320))

    dec_cfg = cfg["decision"]
    unsafe_on = int(dec_cfg.get("unsafe_on_count", 3))
    safe_on = int(dec_cfg.get("safe_on_count", 5))

    fus_cfg = cfg["fusion"]
    fusion = FusionEngine(
        confirm_window_s=float(fus_cfg.get("confirm_window_s", 0.7)),
        handoff_window_s=float(fus_cfg.get("handoff_window_s", 0.7)),
        end_after_s=float(fus_cfg.get("end_after_s", 2.0)),
        cooldown_s=float(fus_cfg.get("cooldown_s", 5.0)),
    )

    alarm_cfg = cfg.get("alarm", {})
    alarm_enabled = bool(alarm_cfg.get("enabled", True))
    alarm = Alarm(
        cooldown_s=float(alarm_cfg.get("cooldown_s", 5.0)),
        audio_path=str(ROOT / alarm_cfg.get("audio_path", "assets/voice_alarm.mp3")),
    )

    fb_cfg = cfg.get("firebase", {})
    firebase_enabled = bool(fb_cfg.get("enabled", False))
    firebase = None

    if firebase_enabled:
        try:
            from firebase_client import FirebaseLogger

            service_path = ROOT / fb_cfg["service_account_path"]
            firebase = FirebaseLogger(
                service_account_path=str(service_path),
                collection=fb_cfg.get("collection", "events"),
            )
            print("[INFO] Firebase startup completed.")

        except Exception as e:
            print(f"[WARN] Firebase startup failed: {e}")
            print("[WARN] Continuing without Firebase.")
            firebase_enabled = False
            firebase = None

    ui_cfg = cfg.get("ui", {})
    show_windows = bool(ui_cfg.get("show_windows", True))
    draw_roi_enabled = bool(ui_cfg.get("draw_roi", True))
    show_fusion = bool(ui_cfg.get("show_fusion_overlay", True))

    cam0 = CameraStream(cam0_id, width=width, height=height)
    cam2 = CameraStream(cam2_id, width=width, height=height)

    roi0 = load_roi(str(roi0_path))
    roi2 = load_roi(str(roi2_path))

    print("ROOT:", ROOT)
    print("ROI0:", roi0_path, "loaded:", roi0 is not None, "points:", 0 if roi0 is None else len(roi0))
    print("ROI2:", roi2_path, "loaded:", roi2 is not None, "points:", 0 if roi2 is None else len(roi2))
    if roi0 is None or roi2 is None:
        print("[ERROR] Missing ROI(s). Run save_roi.py to create ROI files.")
        return

    detector = YoloDetector(model_path=model_path, conf=conf, imgsz=imgsz)
    print("[INFO] YOLO model:", model_path)
    print("[INFO] YOLO imgsz:", imgsz)

    dec0 = SafetyDecision(unsafe_on_count=unsafe_on, safe_on_count=safe_on)
    dec2 = SafetyDecision(unsafe_on_count=unsafe_on, safe_on_count=safe_on)

    print("[INFO] HallGuard main running. Press Q to quit.")

    last_notified_event_id = None

    frame_idx = 0
    last_dets0, last_dets2 = [], []

    infer_last_time = time.time()
    infer_fps_smooth = 0.0

    try:
        while True:
            f0 = cam0.read()
            f2 = cam2.read()
            if f0 is None or f2 is None:
                print("[ERROR] Failed to read from a camera.")
                break

            if draw_roi_enabled:
                draw_roi(f0, roi0)
                draw_roi(f2, roi2)

            crop0, ox0, oy0 = safe_crop(f0, roi0)
            crop2, ox2, oy2 = safe_crop(f2, roi2)

            run_infer = (frame_idx % 2 == 0)

            if run_infer:
                inputs = []
                map_idx = []

                if crop0 is not None:
                    inputs.append(crop0)
                    map_idx.append(0)
                if crop2 is not None:
                    inputs.append(crop2)
                    map_idx.append(1)

                dets0, dets2 = [], []

                if inputs:
                    outs = detector.detect_batch(inputs)
                    for idx, dets in zip(map_idx, outs):
                        if idx == 0:
                            dets0 = dets
                        else:
                            dets2 = dets

                infer_now = time.time()
                infer_dt = infer_now - infer_last_time
                if infer_dt > 0:
                    infer_fps = 1.0 / infer_dt
                    infer_fps_smooth = (
                        infer_fps
                        if infer_fps_smooth == 0.0
                        else (0.9 * infer_fps_smooth + 0.1 * infer_fps)
                    )
                infer_last_time = infer_now

                last_dets0, last_dets2 = dets0, dets2
            else:
                dets0 = last_dets0 if crop0 is not None else []
                dets2 = last_dets2 if crop2 is not None else []

            frame_idx += 1

            unsafe0_seen = has_person(dets0, person_class_id=person_class_id)
            unsafe2_seen = has_person(dets2, person_class_id=person_class_id)

            status0 = dec0.update(unsafe0_seen)
            status2 = dec2.update(unsafe2_seen)

            fused_event, fused_status = fusion.update(status0, status2)

            if alarm_enabled and fused_status == "started":
                alarm.trigger()

            if (
                firebase_enabled
                and firebase is not None
                and fused_status in ("started", "ended")
                and fused_event is not None
            ):
                payload = {
                    "event_id": fused_event.get("event_id"),
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

                doc_id = f"event_{payload['event_id']}"

                try:
                    firebase.log_event(payload, doc_id=doc_id)
                    print("[FIREBASE] queued log:", doc_id, fused_status)

                    import datetime
                    from datetime import timezone

                    now_iso = datetime.datetime.now(timezone.utc).isoformat()

                    live_status = {
                        "state": "UNSAFE" if fused_status == "started" else "SAFE",
                        "raw_status": fused_status,
                        "event_id": payload["event_id"],
                        "location_id": payload["location_id"],
                        "cameras_seen": payload.get("cameras_seen", []),
                        "confirmed_dual": payload.get("confirmed_dual", False),
                        "handoff": payload.get("handoff", False),
                        "client_time": payload.get("client_time", time.time()),
                        "updated_at": now_iso,
                        "online_camera_count": 2,
                        "total_camera_count": 2,
                    }

                    firebase.set_doc("status", "current", live_status, merge=True)
                    print("[FIREBASE] queued status/current:", live_status["state"])

                    if fused_status == "started":
                        current_event_id = str(payload["event_id"])
                        if current_event_id != last_notified_event_id:
                            firebase.send_topic_notification(
                                topic="hallguard_alerts",
                                title="Unsafe event detected",
                                body=f"HallGuard alert at {location_id}",
                                data={
                                    "type": "unsafe_event_started",
                                    "event_id": current_event_id,
                                    "location_id": location_id,
                                    "status": "started",
                                    "title": "Unsafe event detected",
                                    "body": f"HallGuard alert at {location_id}",
                                },
                            )
                            last_notified_event_id = current_event_id
                            print("[FIREBASE] queued FCM notification:", current_event_id)

                    elif fused_status == "ended":
                        ended_event_id = str(payload["event_id"])
                        if ended_event_id == last_notified_event_id:
                            last_notified_event_id = None

                except Exception as e:
                    print(f"[WARN] Firebase queue failed: {e}")

            cv2.putText(
                f0,
                "HallGuard Live",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                f2,
                "HallGuard Live",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                f0,
                f"Infer FPS: {infer_fps_smooth:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                f2,
                f"Infer FPS: {infer_fps_smooth:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                f0,
                f"unsafe_in_roi: {unsafe0_seen}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                f2,
                f"unsafe_in_roi: {unsafe2_seen}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            draw_status(f0, status0, y=125)
            draw_status(f2, status2, y=125)

            if show_fusion:
                draw_fusion_overlay(f0, fused_event, y=160)
                draw_fusion_overlay(f2, fused_event, y=160)

            print(
                f"\rInfer FPS: {infer_fps_smooth:.1f} | imgsz={imgsz} | infer={'YES' if run_infer else 'NO '}",
                end="",
                flush=True,
            )

            if show_windows:
                if f0.shape[0] != f2.shape[0]:
                    target_h = min(f0.shape[0], f2.shape[0])

                    scale0 = target_h / f0.shape[0]
                    scale2 = target_h / f2.shape[0]

                    f0_disp = cv2.resize(f0, (int(f0.shape[1] * scale0), target_h))
                    f2_disp = cv2.resize(f2, (int(f2.shape[1] * scale2), target_h))
                else:
                    f0_disp = f0
                    f2_disp = f2

                combined = cv2.hconcat([f0_disp, f2_disp])

                cv2.imshow("HallGuard - Combined View", combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
            else:
                time.sleep(0.01)

    finally:
        if firebase is not None and hasattr(firebase, "shutdown"):
            firebase.shutdown()

        cam0.release()
        cam2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()