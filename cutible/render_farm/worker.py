"""Render worker — renders a segment of the timeline.

Each worker handles one segment: extract the segment's portion of the
filtergraph, render it, and return the output path.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WorkerStatus(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"


@dataclass
class SegmentTask:
    """A render task for a specific timeline segment."""

    task_id: str
    segment_start: float
    segment_end: float
    project_json: str
    output_path: str
    segment_index: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "segment_index": self.segment_index,
            "output_path": self.output_path,
        }


@dataclass
class SegmentResult:
    """Result of rendering a segment."""

    task_id: str
    success: bool
    output_path: str = ""
    duration: float = 0.0
    size_bytes: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "success": self.success,
            "output_path": self.output_path,
            "duration": self.duration,
        }
        if self.error:
            d["error"] = self.error
        return d


class RenderWorker:
    """Renders a single segment of the timeline.

    Can run locally (subprocess) or be extended for remote execution.
    """

    def __init__(self, worker_id: str = "local"):
        self.worker_id = worker_id
        self.status = WorkerStatus.IDLE
        self._current_task: SegmentTask | None = None

    def render_segment(self, task: SegmentTask) -> SegmentResult:
        """Render a segment of the timeline."""
        self.status = WorkerStatus.BUSY
        self._current_task = task

        try:
            from ..compiler import FFmpegCompiler
            from ..schema import Project

            project = Project.model_validate_json(task.project_json)
            compiler = FFmpegCompiler(project)

            os.makedirs(os.path.dirname(os.path.abspath(task.output_path)), exist_ok=True)

            compiled = compiler.build()
            rs = project.render_settings
            cmd = self._build_segment_command(
                compiled,
                rs,
                task.segment_start,
                task.segment_end,
                task.output_path,
            )

            logger.info(
                f"Worker {self.worker_id}: rendering segment "
                f"{task.segment_index} [{task.segment_start:.1f}s - "
                f"{task.segment_end:.1f}s]"
            )

            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace"
            )
            if proc.returncode != 0:
                tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
                self.status = WorkerStatus.FAILED
                return SegmentResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"ffmpeg failed: {tail}",
                )

            size = os.path.getsize(task.output_path) if os.path.exists(task.output_path) else 0
            result = SegmentResult(
                task_id=task.task_id,
                success=True,
                output_path=task.output_path,
                duration=task.segment_end - task.segment_start,
                size_bytes=size,
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
                error=str(e),
            )

    def _build_segment_command(
        self, compiled, rs, start: float, end: float, output: str
    ) -> list[str]:
        """Build ffmpeg command for a specific segment."""
        cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin"]
        for inp in compiled.inputs:
            cmd += inp
        cmd += ["-filter_complex", compiled.filter_complex]
        for m in compiled.maps:
            cmd += ["-map", m]
        duration = end - start
        cmd += [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{duration:.6f}",
            "-r",
            str(compiled.fps),
            "-c:v",
            rs.vcodec,
            "-preset",
            rs.preset,
            "-crf",
            str(rs.crf),
            "-pix_fmt",
            rs.pix_fmt,
        ]
        if compiled.has_audio:
            cmd += ["-c:a", rs.acodec, "-b:a", rs.audio_bitrate, "-ar", str(rs.audio_rate)]
        cmd += [
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            output,
        ]
        return cmd

    def get_state(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "current_task": self._current_task.to_dict() if self._current_task else None,
        }
