"""Distributed render farm (plan §6.2).

Splits the timeline into segments, distributes rendering across workers,
and assembles the final output. Supports both local subprocess and
remote worker backends.
"""

from .manager import RenderFarmManager
from .scheduler import TaskScheduler
from .worker import RenderWorker

__all__ = ["RenderFarmManager", "RenderWorker", "TaskScheduler"]
