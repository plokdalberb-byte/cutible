"""Multi-agent swarm — specialized agents collaborating on the edit (plan §7).

Each agent has a focused responsibility and communicates through
structured messages. The orchestrator coordinates the workflow.
"""

from .base import BaseAgent, AgentMessage
from .planner import PlannerAgent
from .editor import EditorAgent
from .sound import SoundAgent
from .qc_agent import QCAgent
from .orchestrator import Orchestrator

__all__ = [
    "BaseAgent", "AgentMessage", "PlannerAgent", "EditorAgent",
    "SoundAgent", "QCAgent", "Orchestrator",
]
