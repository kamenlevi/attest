"""Background jobs for the desktop app: start work, poll progress, fetch the result.

Indexing a 700-page book (or vision-transcribing it, or running a 30-question
eval through two models) takes minutes. Holding the HTTP request open that whole
time means the UI shows nothing and a timeout kills the work. So long actions run
as jobs: POST returns a job id immediately, the work runs in a thread and reports
progress ("embedded 512/970 chunks"), and the UI polls GET /api/jobs/{id} until
the status is "done".

The engine stays UI-agnostic: it just calls an optional `report(message,
current, total)` callback it's handed. This module is the only place that knows
about threads and polling.
"""

from __future__ import annotations

import itertools
import threading
from typing import Callable

Report = Callable[..., None]  # report(message, current=None, total=None)


class JobManager:
    """Owns all running/finished jobs for one app instance."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._counter = itertools.count(1)

    def start(self, kind: str, fn: Callable[[Report], dict]) -> str:
        """Run `fn(report)` in a background thread; return the job id at once."""
        job_id = f"{kind}-{next(self._counter)}"
        job = {"id": job_id, "kind": kind, "status": "running",
               "message": "starting…", "current": None, "total": None, "result": None}
        with self._lock:
            self._jobs[job_id] = job

        def report(message: str, current: int | None = None, total: int | None = None) -> None:
            job["message"] = message
            job["current"] = current
            job["total"] = total

        def run() -> None:
            try:
                job["result"] = fn(report)
            except Exception as exc:  # noqa: BLE001 - surface as a friendly message
                job["result"] = {"error": f"{type(exc).__name__}: {exc}"}
            job["status"] = "done"

        threading.Thread(target=run, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> dict:
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": f"unknown job: {job_id}"}
        return dict(job)
