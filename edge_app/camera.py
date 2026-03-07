import time
import cv2
from picamera2 import Picamera2


class CameraStream:
    def __init__(self, camera_id, width=640, height=480):
        self.camera_id = int(camera_id)

        self.picam = Picamera2(self.camera_id)

        config = self.picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )

        self.picam.configure(config)
        self.picam.start()

        time.sleep(1)

        self.last_time = time.time()
        self.fps = 0.0

    def read(self):
        frame = self.picam.capture_array()

        if frame is None:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if dt > 0:
            self.fps = 1.0 / dt

        return frame

    def release(self):
        try:
            self.picam.stop()
            self.picam.close()
        except:
            pass