
import os
import threading

from .database import Sessionlocal
from .models import CompanyDataSource
from .db_sync import sync_data_source, acquire_sync_lease, release_sync_lease

MAX_REQUEUES = 3
REQUEUE_DELAY_SECONDS = 5.0

_pending: dict[str, int] = {}          
_lock = threading.Lock()
_timer: threading.Timer | None = None
_stop = threading.Event()
_thread: threading.Thread | None = None


def _debounce_seconds() -> float:
    return float(os.getenv("MONGO_STREAM_DEBOUNCE_SECONDS", "2"))

def _schedule_flush(delay: float) -> None:
    global _timer
    with _lock:
        if _timer is not None:
            _timer.cancel()
        _timer = threading.Timer(delay, _flush)
        _timer.daemon = True
        _timer.start()

def _mark(collection_name: str) -> None:
    with _lock:
        _pending.setdefault(collection_name, 0)
    _schedule_flush(_debounce_seconds())

def _flush() -> None:
    with _lock:
        batch = dict(_pending)
        _pending.clear()
    if not batch:
        return

    requeue: list[tuple[str, int]] = []
    db = Sessionlocal()
    try:
        sources = (
            db.query(CompanyDataSource)
            .filter(
                CompanyDataSource.source_type == "mongo",
                CompanyDataSource.collection_name.in_(list(batch)),
                CompanyDataSource.is_active.is_(True),
            )
            .all()
        )
        if not sources:
            print(f"[mongo:stream] no active source for {list(batch)}; ignored")
        for source in sources:
            name = source.collection_name
            if not acquire_sync_lease(db, source):
                attempts = batch.get(name, 0)
                if attempts < MAX_REQUEUES:
                    requeue.append((name, attempts + 1))
                else:
                    print(f"[mongo:stream] gave up requeueing {name}; scheduler will catch it")
                continue
            try:
                summary = sync_data_source(db, source)
                print(f"[mongo:stream] company={source.company_id} {summary}")
            except Exception as e:
                db.rollback()
                print(f"[mongo:stream] sync FAILED {name}: {e}")
            finally:
                release_sync_lease(db, source)
    finally:
        db.close()

    if requeue:
        with _lock:
            for name, n in requeue:
                _pending[name] = max(_pending.get(name, 0), n)
        _schedule_flush(REQUEUE_DELAY_SECONDS)

def _catch_up() -> None:
    """Nudge every active Mongo source once, right after the stream opens."""
    db = Sessionlocal()
    try:
        names = [
            n for (n,) in db.query(CompanyDataSource.collection_name).filter(
                CompanyDataSource.source_type == "mongo",
                CompanyDataSource.is_active.is_(True),
                CompanyDataSource.collection_name.isnot(None),
            ).all()
        ]
    finally:
        db.close()
    for n in names:
        _mark(n)

def _listen() -> None:
    from .mongo_client import get_mongo_db

    pipeline = [{
        "$match": {
            "operationType": {"$in": ["insert", "update", "replace", "delete"]},
            "ns.coll": {"$regex": "^company_"},
        }
    }]

    while not _stop.is_set():
        try:
            with get_mongo_db().watch(pipeline, max_await_time_ms=1000) as stream:
                print("[mongo:stream] listening for changes")
                _catch_up()
                while not _stop.is_set() and stream.alive:
                    change = stream.try_next()
                    if change is None:
                        continue
                    _mark(change["ns"]["coll"])
        except Exception as e:
            if "replica set" in str(e).lower():
                print("[mongo:stream] MongoDB is not a replica set; change streams "
                      "unavailable. Scheduler still handles syncs. Listener stopped.")
                return
            print(f"[mongo:stream] listener error, retrying in 10s: {e}")
            _stop.wait(10)

def start_mongo_listener() -> None:
    global _thread
    if os.getenv("MONGO_ENABLED", "false").lower() != "true":
        return
    if os.getenv("MONGO_CHANGE_STREAM_ENABLED", "false").lower() != "true":
        print("[mongo:stream] disabled (MONGO_CHANGE_STREAM_ENABLED is not true)")
        return
    _stop.clear()
    _thread = threading.Thread(target=_listen, daemon=True, name="mongo-change-stream")
    _thread.start()

def stop_mongo_listener() -> None:
    _stop.set()
    with _lock:
        if _timer is not None:
            _timer.cancel()