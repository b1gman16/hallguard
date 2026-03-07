import cv2
import time


class CameraStream:
    def __init__(self, camera_id: int, width=1280, height=720):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 15)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        self.last_time = time.time()
        self.fps = 0.0

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # FPS calculation
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 1.0 / dt

        return frame

    def release(self):
        self.cap.release()
