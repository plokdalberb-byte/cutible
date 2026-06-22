"""High-level composite verbs — the agent's INTENTIONS (plan §3.1).

These are agent-composites that, under the hood, decompose into low-level
primitives + intelligence. They return diffs just like primitives.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

from .schema import (
    Asset, AssetType, Clip, Project, TextLayer, Track, TrackKind, Transform,
)
from .verbs import Editor, Diff, VerbError


class HighLevelVerbs:
    """High-level editing intentions that compose low-level verbs."""

    def __init__(self, editor: Editor, index_store: Optional[Any] = None):
        self.ed = editor
        self.store = index_store

    def remove_silences(self, track_id: str, silence_threshold: float = 0.5,
                        min_silence: float = 0.5) -> Diff:
        """Remove silent portions from all clips on a track."""
        track = self.ed.project.track(track_id)
        if not track.clips:
            raise VerbError("track has no clips", hint="add clips first")
        original_clips = list(track.clips)
        total_removed = 0.0
        for clip in original_clips:
            asset = self.ed.project.asset(clip.asset)
            if asset.type not in (AssetType.video, AssetType.audio):
                continue
            silences = self._detect_silences(asset.uri or "", min_silence)
            if not silences:
                continue
            clip_duration = clip.duration
            remaining = clip_duration
            for s_start, s_end in silences:
                s_start -= clip.src_in
                s_end -= clip.src_in
                if s_start < 0 or s_end > clip.src_duration:
                    continue
                silence_dur = s_end - s_start
                if silence_dur >= min_silence:
                    remaining -= silence_dur
                    total_removed += silence_dur
            if remaining < clip.src_duration:
                self.ed.trim(clip.id, src_out=clip.src_in + remaining)
        self.ed.diff = Diff(
            "remove_silences",
            changed=[c.id for c in original_clips],
            details={"silences_removed": total_removed, "track": track_id},
        )
        return self.ed.diff

    def reframe_to(self, asset_id: str, target_aspect: str = "9:16",
                   focus: str = "center") -> Diff:
        """Reframe a video asset for a different aspect ratio."""
        asset = self.ed.project.asset(asset_id)
        if asset.type not in (AssetType.video, AssetType.color, AssetType.image):
            raise VerbError("can only reframe video/image/color assets")
        w, h = self.ed.project.width, self.ed.project.height
        if target_aspect == "9:16":
            new_w, new_h = 1080, 1920
        elif target_aspect == "16:9":
            new_w, new_h = 1920, 1080
        elif target_aspect == "1:1":
            new_w, new_h = 1080, 1080
        elif target_aspect == "4:3":
            new_w, new_h = 1440, 1080
        else:
            raise VerbError(f"unsupported aspect: {target_aspect}")
        self.ed.project.width = new_w
        self.ed.project.height = new_h
        self.ed.project.aspect = target_aspect
        scale_x = new_w / w
        scale_y = new_h / h
        scale = max(scale_x, scale_y)
        for track in self.ed.project.tracks:
            for clip in track.clips:
                if clip.asset == asset_id:
                    clip.transform.scale = scale
                    if focus == "top":
                        clip.transform.pos_y = -(new_h - h) // 4
                    elif focus == "bottom":
                        clip.transform.pos_y = (new_h - h) // 4
        self.ed.diff = Diff(
            "reframe_to",
            changed=[asset_id],
            details={"aspect": target_aspect, "scale": scale, "focus": focus},
        )
        return self.ed.diff

    def sync_cuts_to_beat(self, track_id: str, audio_track_id: str,
                          tolerance: float = 0.1) -> Diff:
        """Snap clip transitions to the nearest beat in the audio track."""
        audio_track = self.ed.project.track(audio_track_id)
        if not audio_track.clips:
            raise VerbError("audio track has no clips")
        beats = self._get_beats(audio_track)
        if not beats:
            raise VerbError("no beats detected in audio track")
        track = self.ed.project.track(track_id)
        shifted = []
        for clip in track.clips:
            best_beat = min(beats, key=lambda b: abs(b - clip.timeline_in))
            if abs(best_beat - clip.timeline_in) <= tolerance:
                old = clip.timeline_in
                clip.timeline_in = best_beat
                shifted.append({"clip": clip.id, "from": old, "to": best_beat})
        track.clips.sort(key=lambda c: c.timeline_in)
        self.ed.diff = Diff(
            "sync_cuts_to_beat",
            changed=[c.id for c in track.clips],
            details={"shifted": shifted, "n_beats": len(beats)},
        )
        return self.ed.diff

    def generate_captions(self, track_id: str, style: str = "default") -> Diff:
        """Generate captions from transcript and add them as text layers."""
        if self.store is None:
            raise VerbError("index store required for caption generation")
        track = self.ed.project.track(track_id)
        texts = []
        for idx in self.store.get_all_indices():
            for seg in idx.transcript:
                text_id = f"cap_{idx.asset_id}_{len(texts)}"
                layer = TextLayer(
                    id=text_id,
                    text=seg.text,
                    timeline_in=seg.start,
                    timeline_out=seg.end,
                    **self._caption_style(style),
                )
                texts.append(layer)
                track.texts.append(layer)
        track.texts.sort(key=lambda t: t.timeline_in)
        self.ed.diff = Diff(
            "generate_captions",
            changed=[t.id for t in texts],
            details={"style": style, "n_captions": len(texts)},
        )
        return self.ed.diff

    def b_roll_overlay(self, query: str, track_id: str = "v_main",
                       duration: float = 3.0) -> Diff:
        """Find and overlay B-roll footage matching a query."""
        if self.store is None:
            raise VerbError("index store required for B-roll search")
        results = self.store.search_b_roll(query)
        if not results:
            raise VerbError(f"no B-roll found for '{query}'",
                            hint="try a different query or import more assets")
        best = results[0]
        asset_id = best["asset_id"]
        existing_assets = [a.id for a in self.ed.project.assets]
        if asset_id not in existing_assets:
            asset = self.ed.project.asset(asset_id) if asset_id in existing_assets else None
            if asset is None:
                raise VerbError(f"asset {asset_id!r} not in project",
                                hint="add the asset first with add_asset")
        src_start = best["start"]
        track = self.ed.project.track(track_id)
        timeline_start = track.duration if track.clips else 0.0
        diff = self.ed.add_clip(
            track_id, asset_id,
            src_in=src_start, src_out=src_start + duration,
            timeline_in=timeline_start,
            rationale=f"B-roll: {query}",
        )
        return diff

    def auto_ducking(self, voice_track_id: str, music_track_id: str,
                     duck_level: float = 0.15, attack: float = 0.3,
                     release: float = 0.5) -> Diff:
        """Automatically duck music volume when voice is present."""
        voice_track = self.ed.project.track(voice_track_id)
        music_track = self.ed.project.track(music_track_id)
        duck_ranges = []
        for clip in voice_track.clips:
            duck_ranges.append((clip.timeline_in, clip.timeline_out))
        for clip in music_track.clips:
            clip.volume = duck_level
        self.ed.diff = Diff(
            "auto_ducking",
            changed=[c.id for c in music_track.clips],
            details={
                "voice_track": voice_track_id,
                "music_track": music_track_id,
                "duck_level": duck_level,
                "duck_ranges": duck_ranges,
            },
        )
        return self.ed.diff

    def color_match(self, reference_clip_id: str, target_track_ids: list[str]) -> Diff:
        """Apply color matching from a reference clip to other clips."""
        ref_track, ref_clip = self.ed._find_clip(reference_clip_id)
        changed = []
        for track_id in target_track_ids:
            track = self.ed.project.track(track_id)
            for clip in track.clips:
                clip.transform = ref_clip.transform.model_copy()
                changed.append(clip.id)
        self.ed.diff = Diff(
            "color_match",
            changed=changed,
            details={"reference": reference_clip_id, "targets": target_track_ids},
        )
        return self.ed.diff

    def assemble_from_plan(self, edit_plan: dict) -> Diff:
        """Assemble a cut from a structured edit plan.

        edit_plan format:
        {
            "clips": [
                {"asset": "id", "src_in": 0, "src_out": 5, "track": "v1",
                 "timeline_in": 0, "rationale": "why"},
                ...
            ],
            "texts": [
                {"text": "Hello", "track": "captions", "in": 0, "out": 3},
                ...
            ],
            "audio": [
                {"asset": "music", "track": "music", "volume": 0.2},
                ...
            ]
        }
        """
        all_changed = []
        for clip_plan in edit_plan.get("clips", []):
            diff = self.ed.add_clip(
                track_id=clip_plan.get("track", "v_main"),
                asset=clip_plan["asset"],
                src_in=clip_plan.get("src_in", 0),
                src_out=clip_plan["src_out"],
                timeline_in=clip_plan.get("timeline_in"),
                rationale=clip_plan.get("rationale"),
            )
            all_changed.extend(diff.changed)
        for text_plan in edit_plan.get("texts", []):
            diff = self.ed.add_text_layer(
                track_id=text_plan.get("track", "captions"),
                text=text_plan["text"],
                timeline_in=text_plan["in"],
                timeline_out=text_plan["out"],
            )
            all_changed.extend(diff.changed)
        for audio_plan in edit_plan.get("audio", []):
            diff = self.ed.add_audio(
                asset=audio_plan["asset"],
                track_id=audio_plan.get("track", "music"),
                volume=audio_plan.get("volume", 1.0),
            )
            all_changed.extend(diff.changed)
        self.ed.diff = Diff(
            "assemble_from_plan",
            changed=all_changed,
            details={"plan": edit_plan},
        )
        return self.ed.diff

    def make_short(self, topic: str, duration: float = 60.0,
                   asset_id: Optional[str] = None) -> Diff:
        """Create a short clip from source material based on a topic."""
        if self.store is None:
            raise VerbError("index store required for topic search")
        results = self.store.find_moment(topic)
        if not results:
            raise VerbError(f"no content found for topic '{topic}'")
        target_track = "v_short"
        if not any(t.id == target_track for t in self.ed.project.tracks):
            self.ed.add_track(target_track, "video")
        elapsed = 0.0
        all_changed = []
        for r in results:
            if elapsed >= duration:
                break
            src_start = r.get("start", 0)
            src_end = r.get("end", src_start + 5)
            clip_dur = min(src_end - src_start, duration - elapsed)
            aid = r.get("asset_id", asset_id)
            if aid is None:
                continue
            existing = [a.id for a in self.ed.project.assets]
            if aid not in existing:
                continue
            diff = self.ed.add_clip(
                target_track, aid,
                src_in=src_start, src_out=src_start + clip_dur,
                timeline_in=elapsed,
                rationale=f"short about: {topic}",
            )
            all_changed.extend(diff.changed)
            elapsed += clip_dur
        self.ed.diff = Diff(
            "make_short",
            changed=all_changed,
            details={"topic": topic, "target_duration": duration, "actual_duration": elapsed},
        )
        return self.ed.diff

    def _detect_silences(self, uri: str, min_silence: float) -> list[tuple[float, float]]:
        if self.store:
            asset_idx = None
            for idx in self.store.get_all_indices():
                if idx.uri == uri:
                    asset_idx = idx
                    break
            if asset_idx:
                return asset_idx.silence_ranges
        import subprocess, re
        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-i", uri,
            "-af", f"silencedetect=noise=-40dB:d={min_silence}",
            "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                   encoding="utf-8", errors="replace")
            starts = re.findall(r"silence_start:\s*(\d+\.?\d*)", proc.stderr)
            ends = re.findall(r"silence_end:\s*(\d+\.?\d*)", proc.stderr)
            return [(float(s), float(e)) for s, e in zip(starts, ends)]
        except Exception:
            return []

    def _get_beats(self, track: Track) -> list[float]:
        beats = []
        for clip in track.clips:
            asset = self.ed.project.asset(clip.asset)
            if asset.uri and self.store:
                for idx in self.store.get_all_indices():
                    if idx.uri == asset.uri:
                        beats.extend(idx.beat_times)
                        break
        if not beats:
            for clip in track.clips:
                dur = clip.duration
                interval = 0.5
                t = clip.timeline_in
                while t < clip.timeline_out:
                    beats.append(t)
                    t += interval
        return sorted(set(beats))

    def _caption_style(self, style: str) -> dict:
        styles = {
            "default": {"font_size": 48, "font_color": "white", "box": True, "box_color": "black@0.5"},
            "bold": {"font_size": 64, "font_color": "yellow", "box": True, "box_color": "black@0.7"},
            "minimal": {"font_size": 36, "font_color": "white", "box": False},
            "social": {"font_size": 56, "font_color": "white", "box": True, "box_color": "red@0.8"},
        }
        return styles.get(style, styles["default"])
