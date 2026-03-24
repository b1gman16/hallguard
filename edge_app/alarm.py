import time
import platform
import subprocess
from pathlib import Path


class Alarm:
    def __init__(self, cooldown_s=5.0, audio_path=None):
        self.cooldown_s = cooldown_s
        self._last_trigger = 0.0
        self._is_windows = platform.system().lower() == "windows"
        self.audio_path = str(audio_path) if audio_path else None

        if self._is_windows:
            import winsound
            self._winsound = winsound
        else:
            self._winsound = None

    def trigger(self):
        now = time.time()
        if now - self._last_trigger < self.cooldown_s:
            return False

        self._last_trigger = now

        if self.audio_path and Path(self.audio_path).exists():
            try:
                if self._is_windows:
                    # dev fallback on Windows
                    self._winsound.PlaySound(
                        self.audio_path,
                        self._winsound.SND_FILENAME | self._winsound.SND_ASYNC
                    )
                else:
                    # Raspberry Pi / Linux MP3 playback
                    subprocess.Popen(["mpg123", "-q", self.audio_path])
                return True
            except Exception as e:
                print(f"[WARN] Voice alarm failed: {e}")

        # fallback alarm
        if self._is_windows:
            self._winsound.Beep(1200, 200)
            self._winsound.Beep(1200, 200)
            self._winsound.Beep(900, 250)
        else:
            print("\a", end="", flush=True)

        return True