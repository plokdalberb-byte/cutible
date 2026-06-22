"""Export Cutible Timeline-as-Data to OpenTimelineIO format.

Uses the ``opentimelineio`` library to create proper .otio files
that can be opened in DaVinci Resolve, Premiere, and other
OTIO-compatible NLEs.
"""

from __future__ import annotations

import os

try:
    import opentimelineio as otio

    HAS_OTIO = True
except ImportError:
    HAS_OTIO = False

from ..schema import Clip, Project, Track, TrackKind


class OTIOExporter:
    """Export a Cutible project to OpenTimelineIO (.otio) format.

    Creates a proper OTIO Timeline with schema tags that DaVinci/Premiere
    can read natively.
    """

    def __init__(self, project: Project):
        self.p = project

    def export(self, output_path: str) -> dict:
        """Export the project as an OTIO file."""
        if not HAS_OTIO:
            return self._export_fallback(output_path)

        timeline = self._build_timeline()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        otio.adapters.write_to_file(timeline, output_path)
        return {
            "ok": True,
            "output": output_path,
            "format": "otio",
            "schema_version": str(otio.__version__),
        }

    def _build_timeline(self) -> otio.schema.Timeline:
        """Build an OTIO Timeline from the Cutible project."""
        timeline = otio.schema.Timeline(name=self.p.id)
        timeline.metadata["cutible"] = {
            "fps": self.p.fps,
            "width": self.p.width,
            "height": self.p.height,
            "aspect": self.p.aspect,
            "content_hash": self.p.content_hash(),
        }

        for track in self.p.tracks:
            otio_track = self._export_track(track)
            if otio_track is not None:
                timeline.tracks.append(otio_track)

        return timeline

    def _export_track(self, track: Track) -> otio.schema.Track | None:
        """Convert a Cutible track to an OTIO Track."""
        kind_map = {
            TrackKind.video: otio.schema.TrackKind.Video,
            TrackKind.audio: otio.schema.TrackKind.Audio,
            TrackKind.caption: otio.schema.TrackKind.Video,
        }
        otio_kind = kind_map.get(track.kind, otio.schema.TrackKind.Video)
        otio_track = otio.schema.Track(name=track.id, kind=otio_kind)

        for clip in track.clips:
            otio_clip = self._export_clip(clip)
            otio_track.append(otio_clip)

        for text in track.texts:
            otio_text = self._export_text(text)
            otio_track.append(otio_text)

        return otio_track if len(otio_track) > 0 else None

    def _export_clip(self, clip: Clip) -> otio.schema.Clip:
        """Convert a Cutible clip to an OTIO Clip."""
        asset = self.p.asset(clip.asset)
        rate = self.p.fps

        media_ref = otio.schema.ExternalReference(
            target_url=asset.uri or "",
        )

        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(value=int(clip.src_in * rate), rate=rate),
            duration=otio.opentime.RationalTime(value=int(clip.src_duration * rate), rate=rate),
        )

        otio_clip = otio.schema.Clip(
            name=clip.id,
            media_reference=media_ref,
            source_range=source_range,
        )

        otio_clip.metadata["cutible"] = {
            "clip_id": clip.id,
            "asset_id": clip.asset,
            "timeline_in": clip.timeline_in,
            "speed": clip.speed,
            "volume": clip.volume,
            "transition_in": clip.transition_in,
            "transition_out": clip.transition_out,
            "rationale": clip.rationale,
        }

        return otio_clip

    def _export_text(self, text) -> otio.schema.Clip:
        """Convert a Cutible TextLayer to an OTIO Clip with text metadata."""
        rate = self.p.fps

        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(value=int(text.timeline_in * rate), rate=rate),
            duration=otio.opentime.RationalTime(value=int(text.duration * rate), rate=rate),
        )

        otio_clip = otio.schema.Clip(
            name=text.id,
            source_range=source_range,
        )

        otio_clip.metadata["cutible"] = {
            "type": "text_layer",
            "text": text.text,
            "font_size": text.font_size,
            "font_color": text.font_color,
            "x": text.x,
            "y": text.y,
            "box": getattr(text, "box", True),
            "box_color": getattr(text, "box_color", "black@0.5"),
        }

        return otio_clip

    def _export_fallback(self, output_path: str) -> dict:
        """Fallback export when opentimelineio is not installed."""
        import json

        data = {
            "$schema": "OpenTimelineIO/0.14.0",
            "name": self.p.id,
            "tracks": [],
            "metadata": {
                "cutible": {
                    "fps": self.p.fps,
                    "width": self.p.width,
                    "height": self.p.height,
                },
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {
            "ok": True,
            "output": output_path,
            "format": "otio",
            "warning": "fallback export (install opentimelineio for proper OTIO)",
        }
