import json
from pathlib import Path
import cv2
import numpy as np

def load_roi(path: str):
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < 3:
        return None
    return [(int(pt[0]), int(pt[1])) for pt in data]

def draw_roi(frame, roi, color=(0, 255, 0), thickness=3):
    pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, color, thickness)
    for x, y in roi:
        cv2.circle(frame, (x, y), 5, color, -1)

def mask_to_roi(frame, roi):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(roi, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)

def roi_bbox(roi):
    xs = [p[0] for p in roi]
    ys = [p[1] for p in roi]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return x1, y1, x2, y2

