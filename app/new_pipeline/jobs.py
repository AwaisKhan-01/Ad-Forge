"""Tiny thread-based job manager for long pipeline stages.

FastAPI handlers start a job and return its id immediately; the UI follows
progress over SSE (GET /api/jobs/{id}/events). No external queue — a small
startup doesn't need Celery to render five clips.
"""
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class Job:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "running"            # running | done | failed
        self.result: Any = None
        self.error: Optional[str] = None
        self.events: List[Dict] = []
        self._cond = threading.Condition()

    def emit(self, event: str, **data):
        with self._cond:
            self.events.append({"event": event, "ts": time.time(), **data})
            self._cond.notify_all()

    def finish(self, result: Any):
        with self._cond:
            self.status, self.result = "done", result
            self.events.append({"event": "done", "ts": time.time()})
            self._cond.notify_all()

    def fail(self, error: str):
        with self._cond:
            self.status, self.error = "failed", error
            self.events.append({"event": "failed", "ts": time.time(), "error": error})
            self._cond.notify_all()

    def stream(self, poll: float = 15.0):
        """Yield events as they arrive; ends when the job settles."""
        seen = 0
        while True:
            with self._cond:
                if seen >= len(self.events) and self.status == "running":
                    self._cond.wait(timeout=poll)
                batch, seen = self.events[seen:], len(self.events)
            for ev in batch:
                yield ev
            if self.status != "running" and seen >= len(self.events):
                return


_JOBS: Dict[str, Job] = {}


def start(kind: str, fn: Callable[[Job], Any]) -> Job:
    job = Job(kind)
    _JOBS[job.id] = job

    def _run():
        try:
            job.finish(fn(job))
        except Exception as e:              # noqa: BLE001 — job boundary
            job.fail(str(e) or type(e).__name__)

    threading.Thread(target=_run, daemon=True).start()
    return job


def get(job_id: str) -> Optional[Job]:
    return _JOBS.get(job_id)
