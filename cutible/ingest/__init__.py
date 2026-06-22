"""Ingest Pipeline — turning raw media into the agent's memory (plan §5).

Orchestrates: scene detection, transcription, visual understanding,
audio analysis, and embedding generation.
"""

from .pipeline import IngestPipeline
from .scenes import SceneDetector
from .audio_transcribe import AudioTranscriber
from .vlm import VLMAnalyzer
from .audio_analysis import AudioAnalyzer
from .embeddings import EmbeddingGenerator

__all__ = [
    "IngestPipeline", "SceneDetector", "AudioTranscriber",
    "VLMAnalyzer", "AudioAnalyzer", "EmbeddingGenerator",
]
