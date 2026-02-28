class SafetyDecision:
    """
    Simple temporal smoothing:
    - Switch to UNSAFE if unsafe is seen for unsafe_on_count consecutive frames
    - Switch back to SAFE if unsafe is NOT seen for safe_on_count consecutive frames
    """
    def __init__(self, unsafe_on_count=3, safe_on_count=5):
        self.unsafe_on_count = unsafe_on_count
        self.safe_on_count = safe_on_count

        self._unsafe_streak = 0
        self._safe_streak = 0
        self.state = "SAFE"  # or "UNSAFE"

    def update(self, unsafe_seen: bool) -> str:
        if unsafe_seen:
            self._unsafe_streak += 1
            self._safe_streak = 0
        else:
            self._safe_streak += 1
            self._unsafe_streak = 0

        if self.state == "SAFE" and self._unsafe_streak >= self.unsafe_on_count:
            self.state = "UNSAFE"
        elif self.state == "UNSAFE" and self._safe_streak >= self.safe_on_count:
            self.state = "SAFE"

        return self.state
