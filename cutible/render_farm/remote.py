"""Remote render worker — HTTP-based distributed rendering.

Provides a RemoteWorker that sends render tasks to remote workers
via HTTP, and a WorkerServer that accepts render tasks.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Optional
import urllib.request
import urllib.error

from .worker import RenderWorker, SegmentTask, SegmentResult, WorkerStatus

logger = logging.getLogger(__name__)


class RemoteWorker(RenderWorker):
    """Worker that sends render tasks to a remote HTTP endpoint.

    The remote endpoint must accept POST /render with SegmentTask JSON
    and return SegmentResult JSON.
    """

    def __init__(self, worker_id: str, endpoint: str,
                 timeout: int = 600):
        super().__init__(worker_id)
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.status = WorkerStatus.IDLE

    def render_segment(self, task: SegmentTask) -> SegmentResult:
        """Send render task to remote worker via HTTP."""
        self.status = WorkerStatus.BUSY
        self._current_task = task

        try:
            payload = json.dumps(task.to_dict()).encode()
            url = f"{self.endpoint}/render"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            logger.info(f"Remote worker {self.worker_id}: sending task to {url}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())

            result = SegmentResult(
                task_id=data.get("task_id", task.task_id),
                success=data.get("success", False),
                output_path=data.get("output_path", ""),
                duration=data.get("duration", 0),
                error=data.get("error", ""),
            )
            self.status = WorkerStatus.IDLE
            self._current_task = None
            return result

        except Exception as e:
            self.status = WorkerStatus.FAILED
            self._current_task = None
            return SegmentResult(
                task_id=task.task_id,
                success=False,
                error=f"Remote worker error: {e}",
            )

    def health_check(self) -> bool:
        """Check if the remote worker is reachable."""
        try:
            url = f"{self.endpoint}/health"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


def create_worker_app():
    """Create a simple HTTP server that accepts render tasks.

    Run with: python -m cutible.render_farm.remote
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI required for worker server")

    from fastapi import FastAPI
    app = FastAPI(title="Cutible Render Worker")
    worker = RenderWorker("server_worker")

    @app.get("/health")
    def health():
        return {"status": "ok", "worker": worker.get_state()}

    @app.post("/render")
    def render_task(body: dict):
        task = SegmentTask(
            task_id=body["task_id"],
            segment_start=body["segment_start"],
            segment_end=body["segment_end"],
            project_json=body.get("project_json", "{}"),
            output_path=body.get("output_path", f"/tmp/segment_{body['task_id']}.mp4"),
            segment_index=body.get("segment_index", 0),
        )
        result = worker.render_segment(task)
        return result.to_dict()

    return app


HAS_FASTAPI = False
try:
    import fastapi
    HAS_FASTAPI = True
except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    app = create_worker_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)
