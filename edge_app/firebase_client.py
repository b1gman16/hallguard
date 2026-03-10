from pathlib import Path
from typing import Optional, Dict, Any
import datetime
from datetime import timezone
import threading
import queue

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseLogger:
    def __init__(self, service_account_path: str, collection: str = "events"):
        self.collection = collection
        self.db = None
        self.available = False

        # Keep queue very small so Firebase can never build pressure on the main app.
        self._queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

        path = Path(service_account_path)
        if not path.exists():
            raise FileNotFoundError(f"Firebase service account not found: {path}")

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(path))
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self.available = True
        print("[INFO] Firebase initialized.")

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                self._queue.task_done()
                break

            try:
                if task["kind"] == "log_event":
                    self._log_event_now(task["data"], task.get("doc_id"))
                elif task["kind"] == "set_doc":
                    self._set_doc_now(
                        task["collection"],
                        task["doc_id"],
                        task["data"],
                        task.get("merge", True),
                    )
            except Exception as e:
                print(f"[WARN] Firebase worker write failed: {e}")
            finally:
                self._queue.task_done()

    def _log_event_now(self, data: Dict[str, Any], doc_id: Optional[str] = None):
        data = dict(data)
        data["logged_at"] = datetime.datetime.now(timezone.utc).isoformat()

        col = self.db.collection(self.collection)
        if doc_id:
            col.document(doc_id).set(data, merge=True)
        else:
            col.add(data)

    def _set_doc_now(self, collection: str, doc_id: str, data: dict, merge: bool = True):
        self.db.collection(collection).document(doc_id).set(data, merge=merge)

    def _enqueue_latest(self, task: dict):
        if not self.available:
            return

        try:
            self._queue.put_nowait(task)
            return
        except queue.Full:
            pass

        # Drop one old task, then try once more.
        try:
            old = self._queue.get_nowait()
            if old is not None:
                self._queue.task_done()
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(task)
        except queue.Full:
            # If still full, just drop the new task.
            pass

    def log_event(self, data: Dict[str, Any], doc_id: Optional[str] = None):
        self._enqueue_latest({
            "kind": "log_event",
            "data": data,
            "doc_id": doc_id,
        })

    def set_doc(self, collection: str, doc_id: str, data: dict, merge: bool = True):
        self._enqueue_latest({
            "kind": "set_doc",
            "collection": collection,
            "doc_id": doc_id,
            "data": data,
            "merge": merge,
        })

    def shutdown(self):
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass