"""Multi-agent swarm — specialized agents collaborating on the edit (plan §7).

Each agent has a focused responsibility and communicates through
structured messages. The orchestrator coordinates the workflow.
"""

from .base import AgentMessage, BaseAgent
from .editor import EditorAgent
from .orchestrator import Orchestrator
from .planner import PlannerAgent
from .qc_agent import QCAgent
from .sound import SoundAgent

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "PlannerAgent",
    "EditorAgent",
    "SoundAgent",
    "QCAgent",
    "Orchestrator",
]
