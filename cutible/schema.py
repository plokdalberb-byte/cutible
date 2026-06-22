"""Timeline-as-Data: the declarative, diffable, versionable project format.

This is the core artifact of Cutible. The agent reads it, reasons about it,
mutates it through verbs, and the compiler turns it deterministically into a
video. Every clip carries a stable id and an optional ``rationale`` so the
agent's decisions are auditable (see plan §4).
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class AssetType(str, Enum):
    video = "video"
    audio = "audio"
    image = "image"
    color = "color"  # solid color generator, needs no file


class TrackKind(str, Enum):
    video = "video"
    audio = "audio"
    caption = "caption"


# --------------------------------------------------------------------------- #
# Leaf models
# --------------------------------------------------------------------------- #
class Transform(BaseModel):
    """Geometric transform applied to a clip's video."""

    scale: float = 1.0          # uniform scale multiplier
    pos_x: int = 0              # pixel offset from centered position
    pos_y: int = 0
    crop_w: Optional[int] = None
    crop_h: Optional[int] = None
    crop_x: int = 0
    crop_y: int = 0

    model_config = {"extra": "forbid"}


class TextLayer(BaseModel):
    """A burned-in text element (lower third, caption, title)."""

    id: str
    text: str
    timeline_in: float
    timeline_out: float
    x: str = "(w-text_w)/2"     # ffmpeg drawtext expression
    y: str = "h-th-60"
    font_size: int = 48
    font_color: str = "white"
    box: bool = True
    box_color: str = "black@0.5"

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_span(self) -> "TextLayer":
        if self.timeline_out <= self.timeline_in:
            raise ValueError(
                f"text layer {self.id}: timeline_out ({self.timeline_out}) "
                f"must be > timeline_in ({self.timeline_in})"
            )
        return self

    @property
    def duration(self) -> float:
        return round(self.timeline_out - self.timeline_in, 6)


class Clip(BaseModel):
    """A slice of an asset placed on the timeline.

    ``src_in``/``src_out`` address the source media; ``timeline_in`` is where
    it lands on the track. ``timeline_out`` is derived from duration and speed.
    """

    id: str
    asset: str                  # asset id
    src_in: float = 0.0
    src_out: float
    timeline_in: float = 0.0
    speed: float = 1.0
    volume: float = 1.0
    transform: Transform = Field(default_factory=Transform)
    transition_in: float = 0.0   # crossfade/ fade-in seconds
    transition_out: float = 0.0
    rationale: Optional[str] = None  # why the agent placed this clip (audit)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> "Clip":
        if self.src_out <= self.src_in:
            raise ValueError(
                f"clip {self.id}: src_out ({self.src_out}) must be > "
                f"src_in ({self.src_in})"
            )
        if self.speed <= 0:
            raise ValueError(f"clip {self.id}: speed must be > 0")
        if self.timeline_in < 0:
            raise ValueError(f"clip {self.id}: timeline_in must be >= 0")
        return self

    @property
    def src_duration(self) -> float:
        return round(self.src_out - self.src_in, 6)

    @property
    def duration(self) -> float:
        """Duration on the timeline (source duration adjusted for speed)."""
        return round(self.src_duration / self.speed, 6)

    @property
    def timeline_out(self) -> float:
        return round(self.timeline_in + self.duration, 6)


class Asset(BaseModel):
    id: str
    type: AssetType
    uri: Optional[str] = None    # path to media; None for generated (color)
    duration: Optional[float] = None
    # color generator settings
    color: str = "black"
    index_ref: Optional[str] = None  # pointer into the semantic media index

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check(self) -> "Asset":
        if self.type in (AssetType.video, AssetType.audio, AssetType.image):
            if not self.uri:
                raise ValueError(f"asset {self.id}: uri required for {self.type.value}")
        return self


class Track(BaseModel):
    id: str
    kind: TrackKind
    clips: list[Clip] = Field(default_factory=list)
    texts: list[TextLayer] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @property
    def duration(self) -> float:
        ends = [c.timeline_out for c in self.clips] + [t.timeline_out for t in self.texts]
        return round(max(ends), 6) if ends else 0.0


# --------------------------------------------------------------------------- #
# Global / settings
# --------------------------------------------------------------------------- #
class Globals(BaseModel):
    background: str = "black"
    loudness_target: float = -14.0  # LUFS target
    captions_style: Optional[str] = None

    model_config = {"extra": "forbid"}


class RenderSettings(BaseModel):
    # Pinned for determinism (plan §6.2): same project -> same bytes.
    vcodec: str = "libx264"
    acodec: str = "aac"
    crf: int = 18
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    audio_bitrate: str = "192k"
    audio_rate: int = 48000
    container: str = "mp4"

    model_config = {"extra": "forbid"}


class Provenance(BaseModel):
    agent_run_id: Optional[str] = None
    prompt: Optional[str] = None
    edit_plan_ref: Optional[str] = None
    model_versions: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# Root project
# --------------------------------------------------------------------------- #
class Project(BaseModel):
    id: str
    fps: int = 30
    width: int = 1920
    height: int = 1080
    aspect: str = "16:9"
    assets: list[Asset] = Field(default_factory=list)
    tracks: list[Track] = Field(default_factory=list)
    globals: Globals = Field(default_factory=Globals)
    render_settings: RenderSettings = Field(default_factory=RenderSettings)
    provenance: Provenance = Field(default_factory=Provenance)

    model_config = {"extra": "forbid"}

    # ---- lookups ---------------------------------------------------------- #
    def asset(self, asset_id: str) -> Asset:
        for a in self.assets:
            if a.id == asset_id:
                return a
        raise KeyError(f"no asset with id {asset_id!r}")

    def track(self, track_id: str) -> Track:
        for t in self.tracks:
            if t.id == track_id:
                return t
        raise KeyError(f"no track with id {track_id!r}")

    @model_validator(mode="after")
    def _check_refs(self) -> "Project":
        ids = {a.id for a in self.assets}
        if len(ids) != len(self.assets):
            raise ValueError("duplicate asset ids")
        track_ids = [t.id for t in self.tracks]
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("duplicate track ids")
        for t in self.tracks:
            for c in t.clips:
                if c.asset not in ids:
                    raise ValueError(
                        f"clip {c.id} on track {t.id} references unknown asset {c.asset!r}"
                    )
        return self

    @property
    def duration(self) -> float:
        return round(max([t.duration for t in self.tracks], default=0.0), 6)

    # ---- views (cross-granularity, plan §4.3) ----------------------------- #
    def summary(self) -> dict:
        """Compact 'zoom=summary' view for an LLM's context."""
        return {
            "id": self.id,
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "duration": self.duration,
            "n_assets": len(self.assets),
            "tracks": [
                {"id": t.id, "kind": t.kind.value, "n_clips": len(t.clips),
                 "n_texts": len(t.texts), "duration": t.duration}
                for t in self.tracks
            ],
        }

    def outline(self) -> dict:
        """'zoom=outline' view: clips with timecodes, no transforms/effects."""
        return {
            "id": self.id,
            "duration": self.duration,
            "tracks": [
                {
                    "id": t.id,
                    "kind": t.kind.value,
                    "clips": [
                        {"id": c.id, "asset": c.asset,
                         "in": c.timeline_in, "out": c.timeline_out,
                         "rationale": c.rationale}
                        for c in t.clips
                    ],
                    "texts": [
                        {"id": x.id, "text": x.text,
                         "in": x.timeline_in, "out": x.timeline_out}
                        for x in t.texts
                    ],
                }
                for t in self.tracks
            ],
        }

    # ---- serialization / determinism -------------------------------------- #
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent,
                          ensure_ascii=False, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "Project":
        return cls.model_validate_json(text)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def content_hash(self) -> str:
        """Stable hash of the project for caching / golden-test identity."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                               ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
