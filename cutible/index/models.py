"""Data models for the Semantic Media Index.

Each ingested asset produces an AssetIndex with scenes, shots, transcripts,
visual descriptions, and audio features — all time-aligned. The NarrativeIndex
is the cross-asset summary that gives the agent a high-level understanding.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ShotType(str, Enum):
    wide = "wide"
    medium = "medium"
    close_up = "close_up"
    extreme_close_up = "extreme_close_up"
    over_shoulder = "over_shoulder"
    pov = "pov"
    aerial = "aerial"
    establishing = "establishing"


class VisualDescription(BaseModel):
    """VLM-generated description of a visual segment."""

    timestamp: float
    duration: float
    description: str
    shot_type: ShotType | None = None
    subjects: list[str] = Field(default_factory=list)
    action: str = ""
    emotion: str = ""
    lighting: str = ""
    composition: str = ""
    b_roll_potential: float = 0.0  # 0-1, how suitable for B-roll
    quality_score: float = 0.5  # technical quality 0-1

    model_config = {"extra": "forbid"}


class TranscriptSegment(BaseModel):
    """A single transcribed utterance with timing and speaker."""

    start: float
    end: float
    text: str
    speaker: str = "unknown"
    confidence: float = 1.0
    words: list[dict] = Field(default_factory=list)  # [{word, start, end}]

    model_config = {"extra": "forbid"}

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)


class AudioFeatures(BaseModel):
    """Audio analysis features for a segment."""

    timestamp: float
    duration: float
    rms_energy: float = 0.0
    peak_db: float = 0.0
    is_silence: bool = False
    silence_ratio: float = 0.0  # fraction of segment that is silence
    tempo_bpm: float | None = None
    beat_times: list[float] = Field(default_factory=list)
    spectral_centroid_mean: float = 0.0

    model_config = {"extra": "forbid"}


class SpeakerProfile(BaseModel):
    """Detected speaker with their segments."""

    speaker_id: str
    total_speaking_time: float = 0.0
    segment_count: int = 0
    first_appearance: float = 0.0
    last_appearance: float = 0.0
    label: str = ""  # optional human-readable name

    model_config = {"extra": "forbid"}


class Shot(BaseModel):
    """A single shot (continuous camera take) within a scene."""

    id: str
    asset_id: str
    start: float
    end: float
    scene_id: str = ""
    visual: VisualDescription | None = None
    audio: AudioFeatures | None = None
    transcript_segments: list[TranscriptSegment] = Field(default_factory=list)
    keyframe_uri: str | None = None  # path to extracted keyframe image

    model_config = {"extra": "forbid"}

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)

    @property
    def has_speech(self) -> bool:
        return len(self.transcript_segments) > 0

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.transcript_segments)


class Scene(BaseModel):
    """A semantic scene — group of related shots."""

    id: str
    asset_id: str
    start: float
    end: float
    shots: list[Shot] = Field(default_factory=list)
    summary: str = ""
    topic: str = ""
    emotion: str = ""
    narrative_role: str = ""  # "hook", "setup", "climax", "resolution"

    model_config = {"extra": "forbid"}

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)

    @property
    def full_text(self) -> str:
        return " ".join(s.full_text for s in self.shots)


class AssetIndex(BaseModel):
    """Complete semantic index for a single ingested asset."""

    asset_id: str
    uri: str
    duration: float
    fps: int = 30
    width: int = 1920
    height: int = 1080
    scenes: list[Scene] = Field(default_factory=list)
    speakers: list[SpeakerProfile] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    visual_descriptions: list[VisualDescription] = Field(default_factory=list)
    audio_features: list[AudioFeatures] = Field(default_factory=list)
    beat_times: list[float] = Field(default_factory=list)
    tempo_bpm: float | None = None
    silence_ranges: list[tuple[float, float]] = Field(default_factory=list)
    technical_issues: list[dict] = Field(default_factory=list)
    embedding_refs: list[str] = Field(default_factory=list)  # vector DB IDs

    model_config = {"extra": "forbid"}

    def summary(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "duration": self.duration,
            "n_scenes": len(self.scenes),
            "n_shots": sum(len(s.shots) for s in self.scenes),
            "n_speakers": len(self.speakers),
            "has_transcript": bool(self.transcript),
            "tempo_bpm": self.tempo_bpm,
            "n_silence_ranges": len(self.silence_ranges),
        }

    def shots_in_range(self, start: float, end: float) -> list[Shot]:
        result = []
        for scene in self.scenes:
            for shot in scene.shots:
                if shot.end > start and shot.start < end:
                    result.append(shot)
        return result

    def text_at(self, timestamp: float) -> str:
        for seg in self.transcript:
            if seg.start <= timestamp <= seg.end:
                return seg.text
        return ""

    def speaker_at(self, timestamp: float) -> str:
        for seg in self.transcript:
            if seg.start <= timestamp <= seg.end:
                return seg.speaker
        return "unknown"


class NarrativeIndex(BaseModel):
    """Cross-asset high-level narrative understanding.

    Gives the agent a bird's-eye view of all ingested material.
    """

    project_id: str
    total_duration: float = 0.0
    synopsis: str = ""
    key_moments: list[dict] = Field(default_factory=list)
    speakers: list[SpeakerProfile] = Field(default_factory=list)
    asset_indices: list[AssetIndex] = Field(default_factory=list)
    topic_segments: list[dict] = Field(default_factory=list)
    emotional_arc: list[dict] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    def summary(self) -> dict:
        return {
            "project_id": self.project_id,
            "total_duration": self.total_duration,
            "n_assets": len(self.asset_indices),
            "n_speakers": len(self.speakers),
            "synopsis_preview": self.synopsis[:200] + "..."
            if len(self.synopsis) > 200
            else self.synopsis,
        }

    def to_agent_dict(self) -> dict:
        """Convert to the format expected by editing agents.

        Returns a dict with ``assets`` key — a list of asset summaries that
        planner and editor agents can iterate over to build edit plans.
        """
        assets = []
        for ai in self.asset_indices:
            transcript_preview = " ".join(s.text for s in ai.transcript[:5])[:200]
            scenes_summary = [
                {"id": s.id, "start": s.start, "end": s.end, "summary": s.summary or s.topic}
                for s in ai.scenes[:10]
            ]
            assets.append(
                {
                    "asset_id": ai.asset_id,
                    "uri": ai.uri,
                    "duration": ai.duration,
                    "fps": ai.fps,
                    "width": ai.width,
                    "height": ai.height,
                    "n_scenes": len(ai.scenes),
                    "n_transcript_segments": len(ai.transcript),
                    "transcript_preview": transcript_preview,
                    "speakers": [
                        {
                            "id": sp.speaker_id,
                            "label": sp.label,
                            "speaking_time": sp.total_speaking_time,
                        }
                        for sp in ai.speakers
                    ],
                    "beat_times": ai.beat_times[:50],
                    "tempo_bpm": ai.tempo_bpm,
                    "silence_ranges": ai.silence_ranges,
                    "scenes": scenes_summary,
                    "best_segments": [
                        {"start": s.start, "end": s.end, "text": s.full_text[:100]}
                        for s in ai.scenes
                        if s.full_text.strip()
                    ][:10],
                }
            )

        return {
            "project_id": self.project_id,
            "total_duration": self.total_duration,
            "synopsis": self.synopsis,
            "n_assets": len(assets),
            "speakers": [
                {"id": sp.speaker_id, "label": sp.label, "total_time": sp.total_speaking_time}
                for sp in self.speakers
            ],
            "key_moments": self.key_moments,
            "assets": assets,
        }

    def find_moment(self, query: str) -> list[dict]:
        """Simple keyword search across key moments and transcripts."""
        query_lower = query.lower()
        results = []
        for km in self.key_moments:
            if query_lower in km.get("description", "").lower():
                results.append(km)
        for ai in self.asset_indices:
            for seg in ai.transcript:
                if query_lower in seg.text.lower():
                    results.append(
                        {
                            "asset_id": ai.asset_id,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "speaker": seg.speaker,
                        }
                    )
        return results
