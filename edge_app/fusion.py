import time

class FusionEngine:
    """
    Fuses two camera states into a single event stream.

    - Starts an event if any camera becomes UNSAFE
    - Confirms dual if both cameras seen UNSAFE within confirm_window_s
    - Detects handoff if camera A -> B transition happens within handoff_window_s
    - Ends event after no UNSAFE for end_after_s
    - Cooldown prevents immediate re-triggering
    """
    def __init__(
        self,
        confirm_window_s=0.7,
        handoff_window_s=0.7,
        end_after_s=2.0,
        cooldown_s=5.0,
    ):
        self.confirm_window_s = confirm_window_s
        self.handoff_window_s = handoff_window_s
        self.end_after_s = end_after_s
        self.cooldown_s = cooldown_s

        self.event_id = 0
        self.active = False
        self.active_event = None
        self.last_event_end_time = 0.0

        # Track last times each cam was UNSAFE
        self.last_unsafe_time = {"cam0": 0.0, "cam2": 0.0}

    def _new_event(self, now, cameras_seen):
        self.event_id += 1
        self.active = True
        self.active_event = {
            "event_id": self.event_id,
            "start_time": now,
            "last_update": now,
            "cameras_seen": set(cameras_seen),
            "confirmed_dual": False,
            "handoff": False,
        }
        return self.active_event

    def update(self, cam0_state: str, cam2_state: str):
        """
        Returns (event, event_status) where:
        - event is dict or None
        - event_status in {"none","started","updated","ended"}
        """
        now = time.time()
        cam0_unsafe = (cam0_state == "UNSAFE")
        cam2_unsafe = (cam2_state == "UNSAFE")

        # update last unsafe times
        if cam0_unsafe:
            self.last_unsafe_time["cam0"] = now
        if cam2_unsafe:
            self.last_unsafe_time["cam2"] = now

        any_unsafe = cam0_unsafe or cam2_unsafe

        # cooldown gate for starting new event
        if (not self.active) and any_unsafe:
            if now - self.last_event_end_time < self.cooldown_s:
                return None, "none"  # ignore during cooldown
            cameras = []
            if cam0_unsafe:
                cameras.append("cam0")
            if cam2_unsafe:
                cameras.append("cam2")
            ev = self._new_event(now, cameras)
            return ev, "started"

        # if active, update it
        if self.active:
            ev = self.active_event
            ev["last_update"] = now

            # record cameras seen
            if cam0_unsafe:
                ev["cameras_seen"].add("cam0")
            if cam2_unsafe:
                ev["cameras_seen"].add("cam2")

            # confirm dual if both seen close in time
            if (not ev["confirmed_dual"]) and ("cam0" in ev["cameras_seen"]) and ("cam2" in ev["cameras_seen"]):
                # ensure they were seen within confirm window at least once
                if abs(self.last_unsafe_time["cam0"] - self.last_unsafe_time["cam2"]) <= self.confirm_window_s:
                    ev["confirmed_dual"] = True

            # handoff detection: if one cam was unsafe recently then the other becomes unsafe
            # (simple heuristic based on last_unsafe times)
            t0 = self.last_unsafe_time["cam0"]
            t2 = self.last_unsafe_time["cam2"]
            if abs(t0 - t2) <= self.handoff_window_s and ("cam0" in ev["cameras_seen"]) and ("cam2" in ev["cameras_seen"]):
                ev["handoff"] = True

            # end event if no unsafe recently
            last_any_unsafe = max(self.last_unsafe_time["cam0"], self.last_unsafe_time["cam2"])
            if now - last_any_unsafe > self.end_after_s:
                # end
                self.active = False
                self.last_event_end_time = now
                ended_event = ev
                self.active_event = None
                # convert set to list for display/logging
                ended_event["cameras_seen"] = list(ended_event["cameras_seen"])
                return ended_event, "ended"

            # ongoing update
            ev_copy = ev.copy()
            ev_copy["cameras_seen"] = list(ev_copy["cameras_seen"])
            return ev_copy, "updated"

        return None, "none"
