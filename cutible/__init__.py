"""Cutible — Agent-Native montage engine.

Phase 0 + Phase 1 core: a deterministic, diffable "Timeline-as-Data" project
format, a verb/primitive API the agent calls (its "hands"), a deterministic
Timeline -> FFmpeg compiler (the render engine), a QC engine (its "eyes"),
semantic media index (memory), ingest pipeline, multi-agent swarm, VLM
perception loop, Remotion contour, OTIO bridge, render farm, REST API,
and Python SDK.
"""

from .schema import (
    Asset,
    Clip,
    Globals,
    Project,
    Provenance,
    RenderSettings,
    TextLayer,
    Track,
    Transform,
)
from .verbs import Editor, VerbError
from .verbs_high import HighLevelVerbs

__all__ = [
    "Project",
    "Asset",
    "Track",
    "Clip",
    "TextLayer",
    "Transform",
    "Globals",
    "RenderSettings",
    "Provenance",
    "Editor",
    "VerbError",
    "HighLevelVerbs",
]

__version__ = "1.0.0"
