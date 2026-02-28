from pathlib import Path
from typing import Optional, Dict, Any
import datetime
from datetime import timezone

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseLogger:
    def __init__(self, service_account_path: str, collection: str = "events"):
        self.collection = collection
        self.db = None

        path = Path(service_account_path)
        if not path.exists():
            raise FileNotFoundError(f"Firebase service account not found: {path}")

        # Initialize only once
        if not firebase_admin._apps:
            cred = credentials.Certificate(str(path))
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()


    def log_event(self, data: Dict[str, Any], doc_id: Optional[str] = None):
        """
        Writes an event document to Firestore.
        If doc_id provided, uses that as document ID (useful for updating same event).
        """
        # Add server-friendly timestamp
        data = dict(data)
        data["logged_at"] = datetime.datetime.now(timezone.utc).isoformat()


        col = self.db.collection(self.collection)
        if doc_id:
            col.document(doc_id).set(data, merge=True)
            return doc_id
        else:
            ref = col.add(data)[1]
            return ref.id
        
    def set_doc(self, collection: str, doc_id: str, data: dict, merge: bool = True):
        """Set a document at collection/doc_id."""
        self.db.collection(collection).document(doc_id).set(data, merge=merge)
    
