"""Perception loop — the agent's EYES (plan §3.2).

VLM-based semantic review of rendered video + deterministic QC,
forming the closed feedback loop that makes the agent self-correct.
"""

from .proxy_render import ProxyRenderer
from .vlm_review import VLMReview

__all__ = ["VLMReview", "ProxyRenderer"]
