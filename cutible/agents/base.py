"""Base agent class and message types.

All agents share a common interface for receiving tasks, communicating
results, and maintaining state within the orchestrator's coordination.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    FEEDBACK = "feedback"
    QUERY = "query"
    ERROR = "error"


class MessageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Structured message between agents."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    from_agent: str = ""
    to_agent: str = ""
    type: MessageType = MessageType.TASK
    status: MessageStatus = MessageStatus.PENDING
    content: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reply_to: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "type": self.type.value,
            "status": self.status.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentMessage":
        return cls(
            id=d.get("id", ""),
            from_agent=d.get("from_agent", ""),
            to_agent=d.get("to_agent", ""),
            type=MessageType(d.get("type", "task")),
            status=MessageStatus(d.get("status", "pending")),
            content=d.get("content", {}),
            timestamp=d.get("timestamp", ""),
            reply_to=d.get("reply_to"),
        )


class BaseAgent(ABC):
    """Abstract base class for all agents in the swarm.

    Subclasses implement `execute()` with their specific logic.
    """

    def __init__(self, name: str, role: str, description: str = "",
                 llm_client: Optional[Any] = None):
        self.name = name
        self.role = role
        self.description = description
        self.llm = llm_client
        self.inbox: list[AgentMessage] = []
        self.outbox: list[AgentMessage] = []
        self.state: dict[str, Any] = {}
        self._history: list[dict] = []

    @abstractmethod
    def execute(self, message: AgentMessage) -> AgentMessage:
        """Process an incoming message and return a response."""
        ...

    def receive(self, message: AgentMessage) -> None:
        """Add a message to the inbox."""
        self.inbox.append(message)
        self._log("receive", message)

    def send(self, to_agent: str, msg_type: MessageType,
             content: dict, reply_to: Optional[str] = None) -> AgentMessage:
        """Create and queue an outgoing message."""
        msg = AgentMessage(
            from_agent=self.name,
            to_agent=to_agent,
            type=msg_type,
            content=content,
            reply_to=reply_to,
        )
        self.outbox.append(msg)
        self._log("send", msg)
        return msg

    def process_next(self) -> Optional[AgentMessage]:
        """Process the next message in the inbox."""
        if not self.inbox:
            return None
        message = self.inbox.pop(0)
        message.status = MessageStatus.IN_PROGRESS
        try:
            response = self.execute(message)
            message.status = MessageStatus.COMPLETED
            return response
        except Exception as e:
            message.status = MessageStatus.FAILED
            error_msg = self.send(
                to_agent=message.from_agent,
                msg_type=MessageType.ERROR,
                content={"error": str(e), "original_task": message.content},
                reply_to=message.id,
            )
            return error_msg

    def process_all(self) -> list[AgentMessage]:
        """Process all messages in the inbox."""
        responses = []
        while self.inbox:
            resp = self.process_next()
            if resp:
                responses.append(resp)
        return responses

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox),
            "state": self.state,
            "history_count": len(self._history),
        }

    def _log(self, action: str, message: AgentMessage) -> None:
        entry = {
            "action": action,
            "message_id": message.id,
            "from": message.from_agent,
            "to": message.to_agent,
            "type": message.type.value,
            "timestamp": message.timestamp,
        }
        self._history.append(entry)
        logger.debug(f"[{self.name}] {action}: {message.type.value} "
                     f"from={message.from_agent} to={message.to_agent}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} role={self.role!r}>"
