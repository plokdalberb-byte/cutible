"""Import OpenTimelineIO (.otio) files into Cutible Timeline-as-Data.

Uses the ``opentimelineio`` library to read .otio files from
DaVinci Resolve, Premiere, and other OTIO-compatible NLEs.
"""

from __future__ import annotations

try:
    import opentimelineio as otio

    HAS_OTIO = True
except ImportError:
    HAS_OTIO = False

from ..schema import (
    Asset,
    AssetType,
    Clip,
    Project,
    TextLayer,
    Track,
    TrackKind,
)


class OTIOImporter:
    """Import OpenTimelineIO (.otio) files into Cutible projects.

    Parses real OTIO timelines with proper schema tags and maps them
    to Cutible's Timeline-as-Data format.
    """

    def __init__(self):
        self._asset_counter = 0

    def import_file(self, otio_path: str, project_id: str | None = None) -> Project:
        """Import an .otio file and return a Cutible Project."""
        if not HAS_OTIO:
            return self._import_fallback(otio_path, project_id)

        timeline = otio.adapters.read_from_file(otio_path)
        return self._import_timeline(timeline, project_id, otio_path)

    def _import_timeline(
        self, timeline: otio.schema.Timeline, project_id: str | None = None, source_path: str = ""
    ) -> Project:
        """Import an OTIO Timeline object into a Cutible Project."""
        meta = timeline.metadata.get("cutible", {})
        project = Project(
            id=project_id or f"otio_{timeline.name or 'imported'}",
            fps=meta.get("fps", 30),
            width=meta.get("width", 1920),
            height=meta.get("height", 1080),
            aspect=meta.get("aspect", "16:9"),
        )

        asset_map: dict[str, str] = {}

        for otio_track in timeline.tracks:
            track_result = self._import_track(otio_track, project, asset_map)
            if track_result:
                project.tracks.append(track_result)

        return project

    def _import_track(
        self, otio_track: otio.schema.Track, project: Project, asset_map: dict[str, str]
    ) -> Track | None:
        """Convert an OTIO Track to a Cutible Track."""
        kind_map = {
            otio.schema.TrackKind.Video: TrackKind.video,
            otio.schema.TrackKind.Audio: TrackKind.audio,
        }
        kind = kind_map.get(otio_track.kind, TrackKind.video)
        track = Track(id=otio_track.name or f"track_{len(project.tracks)}", kind=kind)

        for child in otio_track:
            if isinstance(child, otio.schema.Clip):
                clip_meta = child.metadata.get("cutible", {})
                if clip_meta.get("type") == "text_layer":
                    text = self._import_text(child)
                    if text:
                        track.texts.append(text)
                else:
                    clip = self._import_clip(child, project, asset_map, kind)
                    if clip:
                        track.clips.append(clip)
            elif isinstance(child, otio.schema.Gap):
                pass  # Skip gaps

        track.clips.sort(key=lambda c: c.timeline_in)
        track.texts.sort(key=lambda t: t.timeline_in)
        return track if track.clips or track.texts else None

    def _import_clip(
        self,
        otio_clip: otio.schema.Clip,
        project: Project,
        asset_map: dict[str, str],
        kind: TrackKind,
    ) -> Clip | None:
        """Convert an OTIO Clip to a Cutible Clip."""
        meta = otio_clip.metadata.get("cutible", {})
        source_range = otio_clip.source_range
        if source_range is None:
            return None

        rate = source_range.start_time.rate
        src_start = source_range.start_time.value / rate
        src_duration = source_range.duration.value / rate
        src_end = src_start + src_duration

        # Get media reference
        media_ref = otio_clip.media_reference
        uri = ""
        if isinstance(media_ref, otio.schema.ExternalReference):
            uri = media_ref.target_url or ""
        elif isinstance(media_ref, otio.schema.MissingReference):
            uri = ""

        asset_id = meta.get("asset_id", otio_clip.name or f"asset_{self._asset_counter}")

        # Register asset if needed
        if asset_id not in asset_map and asset_id not in [a.id for a in project.assets]:
            asset_type = AssetType.video if kind == TrackKind.video else AssetType.audio
            if uri:
                self._asset_counter += 1
                project.assets.append(
                    Asset(
                        id=asset_id,
                        type=asset_type,
                        uri=uri,
                    )
                )
            asset_map[asset_id] = asset_id

        timeline_in = meta.get("timeline_in", 0.0)
        speed = meta.get("speed", 1.0)
        volume = meta.get("volume", 1.0)

        clip = Clip(
            id=meta.get("clip_id", otio_clip.name or f"clip_{self._asset_counter}"),
            asset=asset_id,
            src_in=src_start,
            src_out=src_end,
            timeline_in=timeline_in,
            speed=speed,
            volume=volume,
            transition_in=meta.get("transition_in", 0.0),
            transition_out=meta.get("transition_out", 0.0),
            rationale=meta.get("rationale"),
        )
        return clip

    def _import_text(self, otio_clip: otio.schema.Clip) -> TextLayer | None:
        """Convert an OTIO Clip with text metadata to a TextLayer."""
        meta = otio_clip.metadata.get("cutible", {})
        source_range = otio_clip.source_range
        if source_range is None:
            return None

        rate = source_range.start_time.rate
        start = source_range.start_time.value / rate
        duration = source_range.duration.value / rate
        end = start + duration

        text = meta.get("text", "")
        if not text:
            return None

        return TextLayer(
            id=meta.get("clip_id", otio_clip.name or "text"),
            text=text,
            timeline_in=start,
            timeline_out=end,
            font_size=meta.get("font_size", 48),
            font_color=meta.get("font_color", "white"),
            x=meta.get("x", "(w-text_w)/2"),
            y=meta.get("y", "h-th-60"),
            box=meta.get("box", True),
            box_color=meta.get("box_color", "black@0.5"),
        )

    def _import_fallback(self, otio_path: str, project_id: str | None = None) -> Project:
        """Fallback import when opentimelineio is not installed."""
        import json

        with open(otio_path, encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name", "imported")
        project = Project(id=project_id or f"otio_{name}")
        for track_data in data.get("tracks", []):
            kind_str = track_data.get("kind", "Video")
            track_id = track_data.get("name", f"track_{len(project.tracks)}")
            kind = TrackKind.video if kind_str == "Video" else TrackKind.audio
            track = Track(id=track_id, kind=kind)
            for clip_data in track_data.get("clips", []):
                clip_meta = clip_data.get("metadata", {}).get("cutible", {})
                if clip_meta.get("type") == "text_layer":
                    text = clip_meta.get("text", "")
                    if not text:
                        continue
                    source_range = clip_data.get("source_range", {})
                    rate = source_range.get("start_time", {}).get("rate", 30)
                    start = source_range.get("start_time", {}).get("value", 0) / rate
                    dur = source_range.get("duration", {}).get("value", 0) / rate
                    text_layer = TextLayer(
                        id=clip_meta.get("clip_id", clip_data.get("name", "text")),
                        text=text,
                        timeline_in=start,
                        timeline_out=start + dur,
                        font_size=clip_meta.get("font_size", 48),
                        font_color=clip_meta.get("font_color", "white"),
                        x=clip_meta.get("x", "(w-text_w)/2"),
                        y=clip_meta.get("y", "h-th-60"),
                    )
                    track.texts.append(text_layer)
                    continue
                source_range = clip_data.get("source_range", {})
                rate = source_range.get("start_time", {}).get("rate", 30)
                src_start = source_range.get("start_time", {}).get("value", 0) / rate
                src_end = source_range.get("end_time", {}).get("value", 0) / rate
                media_ref = clip_data.get("media_reference", {})
                uri = media_ref.get("target_url", "")
                asset_id = clip_meta.get("asset_id", clip_data.get("name", "asset"))
                if uri and asset_id not in [a.id for a in project.assets]:
                    project.assets.append(
                        Asset(
                            id=asset_id,
                            type=AssetType.video,
                            uri=uri,
                        )
                    )
                clip = Clip(
                    id=clip_meta.get("clip_id", clip_data.get("name", "clip")),
                    asset=asset_id,
                    src_in=src_start,
                    src_out=src_end,
                    timeline_in=clip_meta.get("timeline_in", 0.0),
                )
                track.clips.append(clip)
            if track.clips or track.texts:
                project.tracks.append(track)
        return project
