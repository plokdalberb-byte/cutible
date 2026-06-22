"""Task scheduler for the render farm.

Splits the timeline into segments and manages task distribution
across workers with priority queuing and dependency tracking.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .worker import SegmentTask, SegmentResult

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ScheduledTask:
    """A task with scheduling metadata."""

    task: SegmentTask
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    result: Optional[SegmentResult] = None
    attempts: int = 0
    max_attempts: int = 3
    priority: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task.task_id,
            "status": self.status.value,
            "segment_index": self.task.segment_index,
            "assigned_worker": self.assigned_worker,
            "attempts": self.attempts,
            "result": self.result.to_dict() if self.result else None,
        }


class TaskScheduler:
    """Manages task distribution for the render farm.

    Splits timelines into segments, creates tasks, assigns them
    to workers, and tracks completion.
    """

    def __init__(self, n_workers: int = 2, max_segment_duration: float = 30.0):
        self.n_workers = n_workers
        self.max_segment_duration = max_segment_duration
        self.tasks: list[ScheduledTask] = []
        self._pending: deque = deque()
        self._completed: list[ScheduledTask] = []

    def create_tasks(self, project_json: str, total_duration: float,
                     output_dir: str) -> list[ScheduledTask]:
        """Split the timeline into segments and create render tasks."""
        segments = self._split_duration(total_duration)
        tasks = []
        for i, (start, end) in enumerate(segments):
            task_id = f"seg_{uuid.uuid4().hex[:8]}"
            task = SegmentTask(
                task_id=task_id,
                segment_start=start,
                segment_end=end,
                project_json=project_json,
                output_path=f"{output_dir}/segment_{i:04d}.mp4",
                segment_index=i,
            )
            scheduled = ScheduledTask(task=task)
            tasks.append(scheduled)
            self._pending.append(scheduled)
        self.tasks.extend(tasks)
        logger.info(f"Created {len(tasks)} render tasks for {total_duration:.1f}s timeline")
        return tasks

    def get_next_task(self, worker_id: str) -> Optional[ScheduledTask]:
        """Get the next pending task for a worker."""
        while self._pending:
            task = self._pending.popleft()
            if task.attempts < task.max_attempts:
                task.status = TaskStatus.ASSIGNED
                task.assigned_worker = worker_id
                return task
        return None

    def complete_task(self, task_id: str, result: SegmentResult) -> None:
        """Mark a task as completed or failed."""
        for task in self.tasks:
            if task.task.task_id == task_id:
                task.result = result
                if result.success:
                    task.status = TaskStatus.COMPLETED
                    self._completed.append(task)
                else:
                    task.attempts += 1
                    if task.attempts < task.max_attempts:
                        task.status = TaskStatus.PENDING
                        self._pending.append(task)
                        logger.warning(f"Task {task_id} failed, retrying "
                                       f"({task.attempts}/{task.max_attempts})")
                    else:
                        task.status = TaskStatus.FAILED
                        logger.error(f"Task {task_id} failed permanently")
                break

    def get_completed_outputs(self) -> list[str]:
        """Get ordered list of completed segment output paths."""
        completed = sorted(
            [t for t in self.tasks if t.status == TaskStatus.COMPLETED],
            key=lambda t: t.task.segment_index,
        )
        return [t.result.output_path for t in completed if t.result]

    def is_complete(self) -> bool:
        """Check if all tasks are completed or failed."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for t in self.tasks
        )

    def get_progress(self) -> dict:
        """Get current progress summary."""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        running = sum(1 for t in self.tasks if t.status in (TaskStatus.ASSIGNED, TaskStatus.RUNNING))
        pending = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "percent": round(completed / max(total, 1) * 100, 1),
        }

    def _split_duration(self, duration: float) -> list[tuple[float, float]]:
        """Split duration into segments, respecting max_segment_duration."""
        segments = []
        t = 0.0
        while t < duration:
            end = min(t + self.max_segment_duration, duration)
            segments.append((round(t, 6), round(end, 6)))
            t = end
        return segments
