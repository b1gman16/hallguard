from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.4, imgsz=320, person_class_id=0):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.person_class_id = person_class_id

    def detect_batch(self, images):
        results = self.model.predict(
            images,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[self.person_class_id],
            max_det=2,
            verbose=False,
        )

        batch = []
        for r in results:
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    dets.append(
                        {
                            "cls": int(b.cls[0].item()),
                            "conf": float(b.conf[0].item()),
                            "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                        }
                    )
            batch.append(dets)

        return batch