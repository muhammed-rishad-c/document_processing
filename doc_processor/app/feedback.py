import json
import threading
from pathlib import Path
from datetime import datetime, timezone

FEEDBACK_LOG_PATH = Path(__file__).resolve().parent / "feedback_log.jsonl"
_write_lock = threading.Lock()

MAX_COMMENT_LENGTH = 1000


def log_feedback(*, session_id: str, rating: int, comment: str | None = None) -> None:
    
    if not (1 <= rating <= 5):
        raise ValueError("rating must be between 1 and 5")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "rating": rating,
        "comment": (comment or "").strip()[:MAX_COMMENT_LENGTH] or None,
    }

    with _write_lock, open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")