"""Semantic Media Index — the agent's MEMORY (plan §5).

Structured, time-aligned representation of source media that lets the agent
reason about content without watching every frame.
"""

from .models import (
    Scene,
    Shot,
    TranscriptSegment,
    VisualDescription,
    AudioFeatures,
    SpeakerProfile,
    AssetIndex,
    NarrativeIndex,
)
from .store import IndexStore
from .search import IndexSearcher

__all__ = [
    "Scene", "Shot", "TranscriptSegment", "VisualDescription",
    "AudioFeatures", "SpeakerProfile", "AssetIndex", "NarrativeIndex",
    "IndexStore", "IndexSearcher",
]
