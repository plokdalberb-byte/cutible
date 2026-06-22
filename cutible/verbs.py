"""The agent's HANDS — a verb/primitive API over Timeline-as-Data.

Design principles from plan §3.1, made literal:

* Every verb returns a machine-readable result + a diff of what changed, so
  the agent can reason about the next step.
* Clips/layers are addressable by stable id, never by "mouse coordinates".
* Errors are structured and instructive so the agent can self-correct.
* Checkpoints + undo: the agent tries, inspects, and reverts (git-like).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .schema import (
    Asset,
    AssetType,
    Clip,
    Project,
    TextLayer,
    Track,
    TrackKind,
    Transform,
)


class VerbError(ValueError):
    """Structured, instructive error the agent is expected to recover from."""

    def __init__(self, message: str, *, hint: str = "", context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.context = context or {}

    def to_dict(self) -> dict:
        return {"error": self.message, "hint": self.hint, "context": self.context}


@dataclass
class Diff:
    verb: str
    changed: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"verb": self.verb, "changed": self.changed, "details": self.details}


class Editor:
    """Stateful editing session: holds a Project, applies verbs, tracks history.

    Usage::

        ed = Editor(project)
        ed.add_clip("v1", asset="cam1", src_in=0, src_out=5)
        ed.checkpoint("rough cut")
        ed.trim("clip_1", src_in=1.0)
        result = ed.diff   # machine-readable last change
    """

    def __init__(self, project: Project):
        self.project = project
        self._history: list[tuple[str, Project]] = []  # (label, snapshot-before)
        self._counter = 0
        self.diff: Optional[Diff] = None

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _snapshot(self, label: str) -> None:
        self._history.append((label, copy.deepcopy(self.project)))

    def _find_clip(self, clip_id: str) -> tuple[Track, Clip]:
        for t in self.project.tracks:
            for c in t.clips:
                if c.id == clip_id:
                    return t, c
        raise VerbError(
            f"no clip with id {clip_id!r}",
            hint="call read('outline') to see valid clip ids",
            context={"known_clips": [c.id for t in self.project.tracks for c in t.clips]},
        )

    def _validate(self) -> None:
        # Re-validate the whole project; pydantic raises on broken refs/spans.
        self.project = Project.model_validate(self.project.model_dump(mode="json"))

    # ------------------------------------------------------------------ #
    # checkpoint / undo / branch (plan §3.1)
    # ------------------------------------------------------------------ #
    def checkpoint(self, label: str = "") -> Diff:
        self._snapshot(label or f"checkpoint_{len(self._history)}")
        self.diff = Diff("checkpoint", details={"label": label, "depth": len(self._history)})
        return self.diff

    def undo(self) -> Diff:
        if not self._history:
            raise VerbError("nothing to undo", hint="no checkpoints recorded yet")
        label, snap = self._history.pop()
        self.project = snap
        self.diff = Diff("undo", details={"restored": label, "depth": len(self._history)})
        return self.diff

    def branch(self) -> "Editor":
        """Fork an independent editing session (try alternatives in parallel)."""
        return Editor(copy.deepcopy(self.project))

    # ------------------------------------------------------------------ #
    # asset / track management
    # ------------------------------------------------------------------ #
    def add_asset(self, asset_id: str, type: str, uri: Optional[str] = None,
                  duration: Optional[float] = None, color: str = "black",
                  index_ref: Optional[str] = None) -> Diff:
        if any(a.id == asset_id for a in self.project.assets):
            raise VerbError(f"asset id {asset_id!r} already exists",
                            hint="use a unique asset id")
        a = Asset(id=asset_id, type=AssetType(type), uri=uri, duration=duration,
                  color=color, index_ref=index_ref)
        self.project.assets.append(a)
        self.diff = Diff("add_asset", changed=[asset_id], details=a.model_dump(mode="json"))
        return self.diff

    def add_track(self, track_id: str, kind: str) -> Diff:
        if any(t.id == track_id for t in self.project.tracks):
            raise VerbError(f"track id {track_id!r} already exists")
        self.project.tracks.append(Track(id=track_id, kind=TrackKind(kind)))
        self.diff = Diff("add_track", changed=[track_id], details={"kind": kind})
        return self.diff

    # ------------------------------------------------------------------ #
    # PRIMITIVES (plan §3.1 low-level verbs)
    # ------------------------------------------------------------------ #
    def add_clip(self, track_id: str, asset: str, src_in: float = 0.0,
                 src_out: Optional[float] = None, timeline_in: Optional[float] = None,
                 speed: float = 1.0, volume: float = 1.0,
                 clip_id: Optional[str] = None, rationale: Optional[str] = None,
                 append: bool = True) -> Diff:
        """Place a slice of an asset on a track."""
        try:
            track = self.project.track(track_id)
        except KeyError as e:
            raise VerbError(str(e), hint="add the track first with add_track") from e
        try:
            asset_obj = self.project.asset(asset)
        except KeyError as e:
            raise VerbError(str(e), hint="add the asset first with add_asset",
                            context={"known_assets": [a.id for a in self.project.assets]}) from e

        if src_out is None:
            if asset_obj.duration is None:
                raise VerbError(
                    f"src_out omitted but asset {asset!r} has no known duration",
                    hint="pass src_out explicitly or set the asset's duration")
            src_out = asset_obj.duration

        if timeline_in is None:
            timeline_in = track.duration if append else 0.0

        clip_id = clip_id or self._next_id("clip")
        clip = Clip(id=clip_id, asset=asset, src_in=src_in, src_out=src_out,
                    timeline_in=timeline_in, speed=speed, volume=volume,
                    rationale=rationale)
        track.clips.append(clip)
        track.clips.sort(key=lambda c: c.timeline_in)
        self._validate()
        self.diff = Diff("add_clip", changed=[clip_id],
                         details={"track": track_id, "timeline_in": timeline_in,
                                  "timeline_out": clip.timeline_out,
                                  "duration": clip.duration})
        return self.diff

    def trim(self, clip_id: str, src_in: Optional[float] = None,
             src_out: Optional[float] = None) -> Diff:
        """Adjust a clip's source in/out points."""
        _, clip = self._find_clip(clip_id)
        before = clip.duration
        new_in = clip.src_in if src_in is None else src_in
        new_out = clip.src_out if src_out is None else src_out
        if new_out <= new_in:
            raise VerbError(
                f"trim would make clip {clip_id} non-positive: in={new_in} out={new_out}",
                hint="ensure src_out > src_in")
        clip.src_in, clip.src_out = new_in, new_out
        self._validate()
        self.diff = Diff("trim", changed=[clip_id],
                         details={"src_in": new_in, "src_out": new_out,
                                  "duration_before": before, "duration_after": clip.duration})
        return self.diff

    def move(self, clip_id: str, timeline_in: float) -> Diff:
        """Move a clip to a new position on its track."""
        track, clip = self._find_clip(clip_id)
        if timeline_in < 0:
            raise VerbError("timeline_in must be >= 0", hint="use a non-negative time")
        old = clip.timeline_in
        clip.timeline_in = timeline_in
        track.clips.sort(key=lambda c: c.timeline_in)
        self.diff = Diff("move", changed=[clip_id],
                         details={"from": old, "to": timeline_in})
        return self.diff

    def split(self, clip_id: str, t: float) -> Diff:
        """Split a clip at timeline time ``t`` into two addressable clips."""
        track, clip = self._find_clip(clip_id)
        if not (clip.timeline_in < t < clip.timeline_out):
            raise VerbError(
                f"split time {t} outside clip {clip_id} "
                f"[{clip.timeline_in}, {clip.timeline_out}]",
                hint="choose t strictly inside the clip span")
        rel = (t - clip.timeline_in) * clip.speed   # source-time offset
        cut_src = round(clip.src_in + rel, 6)
        right_id = self._next_id("clip")
        right = Clip(id=right_id, asset=clip.asset, src_in=cut_src,
                     src_out=clip.src_out, timeline_in=t, speed=clip.speed,
                     volume=clip.volume, rationale=clip.rationale)
        clip.src_out = cut_src
        track.clips.append(right)
        track.clips.sort(key=lambda c: c.timeline_in)
        self._validate()
        self.diff = Diff("split", changed=[clip_id, right_id],
                         details={"at": t, "left": clip_id, "right": right_id})
        return self.diff

    def ripple_delete(self, clip_id: str) -> Diff:
        """Delete a clip and pull subsequent clips on the track left to close the gap."""
        track, clip = self._find_clip(clip_id)
        gap = clip.duration
        start = clip.timeline_in
        track.clips = [c for c in track.clips if c.id != clip_id]
        shifted = []
        for c in track.clips:
            if c.timeline_in >= start:
                c.timeline_in = round(c.timeline_in - gap, 6)
                shifted.append(c.id)
        track.clips.sort(key=lambda c: c.timeline_in)
        self.diff = Diff("ripple_delete", changed=[clip_id] + shifted,
                         details={"removed": clip_id, "closed_gap": gap, "shifted": shifted})
        return self.diff

    def set_speed(self, clip_id: str, speed: float) -> Diff:
        _, clip = self._find_clip(clip_id)
        if speed <= 0:
            raise VerbError("speed must be > 0", hint="e.g. 0.5 = slow-mo, 2.0 = fast")
        old = clip.speed
        clip.speed = speed
        self.diff = Diff("set_speed", changed=[clip_id],
                         details={"from": old, "to": speed, "new_duration": clip.duration})
        return self.diff

    def set_volume(self, clip_id: str, volume: float) -> Diff:
        _, clip = self._find_clip(clip_id)
        clip.volume = volume
        self.diff = Diff("set_volume", changed=[clip_id], details={"volume": volume})
        return self.diff

    def add_transition(self, clip_id: str, kind: str = "in",
                       duration: float = 0.5) -> Diff:
        """Add a fade/crossfade of ``duration`` seconds to a clip edge."""
        _, clip = self._find_clip(clip_id)
        if kind not in ("in", "out"):
            raise VerbError("kind must be 'in' or 'out'")
        if duration < 0 or duration > clip.duration:
            raise VerbError(
                f"transition {duration}s does not fit clip duration {clip.duration}s",
                hint="shorten the transition or lengthen the clip")
        if kind == "in":
            clip.transition_in = duration
        else:
            clip.transition_out = duration
        self.diff = Diff("add_transition", changed=[clip_id],
                         details={"kind": kind, "duration": duration})
        return self.diff

    def add_text_layer(self, track_id: str, text: str, timeline_in: float,
                       timeline_out: float, text_id: Optional[str] = None,
                       **style) -> Diff:
        """Add a burned-in caption / title to a caption (or video) track."""
        try:
            track = self.project.track(track_id)
        except KeyError as e:
            raise VerbError(str(e), hint="add a caption track first") from e
        text_id = text_id or self._next_id("text")
        layer = TextLayer(id=text_id, text=text, timeline_in=timeline_in,
                          timeline_out=timeline_out, **style)
        track.texts.append(layer)
        track.texts.sort(key=lambda x: x.timeline_in)
        self.diff = Diff("add_text_layer", changed=[text_id],
                         details={"track": track_id, "in": timeline_in, "out": timeline_out})
        return self.diff

    def add_audio(self, asset: str, src_in: float = 0.0,
                  src_out: Optional[float] = None, timeline_in: float = 0.0,
                  volume: float = 1.0, track_id: str = "music",
                  rationale: Optional[str] = None) -> Diff:
        """Convenience: ensure an audio track exists and drop an audio clip on it."""
        if not any(t.id == track_id for t in self.project.tracks):
            self.add_track(track_id, "audio")
        return self.add_clip(track_id, asset=asset, src_in=src_in, src_out=src_out,
                             timeline_in=timeline_in, volume=volume, rationale=rationale)

    # ------------------------------------------------------------------ #
    # READ — the agent inspects its own work (plan §3.2 readable state)
    # ------------------------------------------------------------------ #
    def read(self, zoom: str = "outline") -> dict:
        if zoom == "summary":
            return self.project.summary()
        if zoom == "outline":
            return self.project.outline()
        if zoom == "detail":
            return self.project.model_dump(mode="json")
        raise VerbError(f"unknown zoom {zoom!r}",
                        hint="use 'summary', 'outline', or 'detail'")

    def __repr__(self) -> str:
        return f"<Editor project={self.project.id!r} duration={self.project.duration}s>"
