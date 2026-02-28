import time
import platform

class Alarm:
    def __init__(self, cooldown_s=5.0):
        self.cooldown_s = cooldown_s
        self._last_trigger = 0.0

        self._is_windows = (platform.system().lower() == "windows")
        if self._is_windows:
            import winsound
            self._winsound = winsound
        else:
            self._winsound = None

    def trigger(self):
        """Trigger alarm if not in cooldown."""
        now = time.time()
        if now - self._last_trigger < self.cooldown_s:
            return False
        self._last_trigger = now

        # Laptop dev alarm
        if self._is_windows:
            # frequency, duration(ms)
            self._winsound.Beep(1200, 200)
            self._winsound.Beep(1200, 200)
            self._winsound.Beep(900, 250)
        else:
            # fallback for non-windows
            print("\a", end="", flush=True)

        return True
