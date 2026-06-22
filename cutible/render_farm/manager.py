"""Render farm manager — orchestrates distributed rendering.

Coordinates multiple workers, manages the task queue, and assembles
segmented outputs into the final rendered video.
"""

from __future__ import annotations

import logging
import os
import subprocess

from ..compiler import FFmpegCompiler
from ..schema import Project
from .scheduler import TaskScheduler
from .worker import RenderWorker, SegmentResult

logger = logging.getLogger(__name__)


class RenderFarmManager:
    """Manages distributed rendering across multiple workers.

    Splits the timeline, distributes segments to workers,
    collects results, and assembles the final output.

    Supports both local and remote workers:
    - Local: workers render via subprocess (default)
    - Remote: workers receive HTTP POST /render tasks
    """

    def __init__(
        self,
        n_workers: int = 2,
        max_segment_duration: float = 30.0,
        output_dir: str = ".cutible/render_farm",
        remote_endpoints: list[str] | None = None,
    ):
        self.n_workers = n_workers
        self.output_dir = output_dir
        self.scheduler = TaskScheduler(n_workers, max_segment_duration)

        if remote_endpoints:
            from .remote import RemoteWorker

            self.workers = [
                RemoteWorker(f"remote_{i}", endpoint) for i, endpoint in enumerate(remote_endpoints)
            ]
        else:
            self.workers = [RenderWorker(f"worker_{i}") for i in range(n_workers)]

        os.makedirs(output_dir, exist_ok=True)

    def render(self, project: Project, output_path: str, parallel: bool = True) -> dict:
        """Render the project using the distributed farm."""
        compiler = FFmpegCompiler(project)
        compiled = compiler.build()

        if compiled.duration <= 0:
            return {"ok": False, "error": "project has zero duration"}

        project_json = project.model_dump_json()
        self.scheduler = TaskScheduler(
            self.n_workers,
            max_segment_duration=30.0,
        )
        tasks = self.scheduler.create_tasks(
            project_json,
            compiled.duration,
            self.output_dir,
        )

        if not tasks:
            return {"ok": False, "error": "no segments created"}

        results = self._render_parallel() if parallel else self._render_sequential()

        completed = self.scheduler.get_completed_outputs()
        if not completed:
            failed = [r for r in results if not r.success]
            return {
                "ok": False,
                "error": "all segments failed",
                "failures": [r.to_dict() for r in failed],
            }

        assembly_result = self._assemble_segments(completed, output_path)

        return {
            "ok": assembly_result.get("ok", False),
            "output": output_path,
            "segments": len(completed),
            "total_segments": len(tasks),
            "duration": compiled.duration,
            "size_bytes": os.path.getsize(output_path) if os.path.exists(output_path) else 0,
            "progress": self.scheduler.get_progress(),
        }

    def render_dry_run(self, project: Project) -> dict:
        """Show what the render farm would do without actually rendering."""
        compiler = FFmpegCompiler(project)
        compiled = compiler.build()
        segments = self.scheduler._split_duration(compiled.duration)
        return {
            "duration": compiled.duration,
            "n_segments": len(segments),
            "segments": [{"start": s, "end": e} for s, e in segments],
            "n_workers": self.n_workers,
        }

    def _render_parallel(self) -> list[SegmentResult]:
        """Render segments in parallel across workers."""
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for worker in self.workers:
                task = self.scheduler.get_next_task(worker.worker_id)
                if task:
                    task.status = "running"
                    futures[executor.submit(worker.render_segment, task.task)] = (worker, task)

            while futures:
                done, _ = concurrent.futures.wait(
                    futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    worker, task = futures.pop(future)
                    try:
                        result = future.result()
                        self.scheduler.complete_task(task.task.task_id, result)
                        results.append(result)
                        next_task = self.scheduler.get_next_task(worker.worker_id)
                        if next_task:
                            next_task.status = "running"
                            futures[executor.submit(worker.render_segment, next_task.task)] = (
                                worker,
                                next_task,
                            )
                    except Exception as e:
                        self.scheduler.complete_task(
                            task.task.task_id,
                            SegmentResult(task_id=task.task.task_id, success=False, error=str(e)),
                        )
        return results

    def _render_sequential(self) -> list[SegmentResult]:
        """Render segments sequentially with a single worker."""
        results = []
        worker = self.workers[0]
        while True:
            task = self.scheduler.get_next_task(worker.worker_id)
            if task is None:
                break
            result = worker.render_segment(task.task)
            self.scheduler.complete_task(task.task.task_id, result)
            results.append(result)
        return results

    def _assemble_segments(self, segment_paths: list[str], output_path: str) -> dict:
        """Concatenate rendered segments into the final output."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        concat_file = os.path.join(self.output_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for path in segment_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-nostdin",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace"
        )
        if proc.returncode != 0:
            return {"ok": False, "error": f"assembly failed: {proc.stderr[-200:]}"}
        return {"ok": True}

    def get_state(self) -> dict:
        return {
            "n_workers": self.n_workers,
            "workers": [w.get_state() for w in self.workers],
            "progress": self.scheduler.get_progress(),
        }
