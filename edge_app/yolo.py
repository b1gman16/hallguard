from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.4):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image_bgr):
        """
        Returns a list of detections:
        [{'cls': int, 'conf': float, 'xyxy': (x1,y1,x2,y2)}, ...]
        """
        results = self.model.predict(image_bgr, conf=self.conf, verbose=False)
        r = results[0]

        dets = []
        if r.boxes is None:
            return dets

        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item())
            cls = int(b.cls[0].item())
            dets.append({
                "cls": cls,
                "conf": conf,
                "xyxy": (int(x1), int(y1), int(x2), int(y2)),
            })
        return dets
