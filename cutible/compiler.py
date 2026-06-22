"""Deterministic render engine — Timeline-as-Data -> FFmpeg filtergraph.

This is "Contour A" from plan §6.1: a compiler that turns the declarative
project into a single ``ffmpeg`` invocation (filter_complex) and runs it.
Same project in -> same video out (versions pinned in RenderSettings).

The compiler builds, for every clip, one ffmpeg input, then assembles:
  * a video filter chain (trim -> speed -> fit/crop -> fade -> time-offset)
    overlaid onto a solid canvas, track by track, time by time;
  * burned-in drawtext layers on top;
  * an audio graph (atrim -> tempo -> volume -> fade -> delay -> amix -> loudnorm).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from .schema import Asset, AssetType, Clip, Project, TextLayer, Track, TrackKind


def _esc_drawtext(text: str) -> str:
    """Escape text for ffmpeg drawtext."""
    out = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")
    out = out.replace("%", "\\%")
    return out


@dataclass
class CompiledRender:
    inputs: list[list[str]]
    filter_complex: str
    maps: list[str]
    has_audio: bool
    duration: float

    fps: int = 30

    def command(self, out_path: str, rs) -> list[str]:
        cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-nostdin"]
        for inp in self.inputs:
            cmd += inp
        cmd += ["-filter_complex", self.filter_complex]
        for m in self.maps:
            cmd += ["-map", m]
        cmd += [
            "-r", str(self.fps),
            "-c:v", rs.vcodec, "-preset", rs.preset, "-crf", str(rs.crf),
            "-pix_fmt", rs.pix_fmt,
        ]
        if self.has_audio:
            cmd += ["-c:a", rs.acodec, "-b:a", rs.audio_bitrate, "-ar", str(rs.audio_rate)]
        cmd += [
            "-t", f"{self.duration:.6f}",
            "-map_metadata", "-1",
            "-movflags", "+faststart",
            out_path,
        ]
        return cmd


class FFmpegCompiler:
    def __init__(self, project: Project):
        self.p = project

    # ------------------------------------------------------------------ #
    def _input_args(self, clip: Clip, asset: Asset, kind: TrackKind) -> list[str]:
        """ffmpeg input args for one clip (one input per clip)."""
        if asset.type == AssetType.color:
            return ["-f", "lavfi", "-t", f"{clip.duration:.6f}",
                    "-i", f"color=c={asset.color}:s={self.p.width}x{self.p.height}:"
                          f"r={self.p.fps}"]
        if asset.type == AssetType.image:
            return ["-loop", "1", "-t", f"{clip.duration:.6f}",
                    "-i", asset.uri]
        # video / audio file
        return ["-i", asset.uri]

    # ------------------------------------------------------------------ #
    def _video_chain(self, idx: int, clip: Clip, asset: Asset, out_label: str) -> str:
        W, H, FPS = self.p.width, self.p.height, self.p.fps
        steps: list[str] = []
        src = f"[{idx}:v]"
        steps.append("settb=AVTB")

        if asset.type == AssetType.video:
            steps.append(f"trim=start={clip.src_in:.6f}:end={clip.src_out:.6f}")
            steps.append("setpts=PTS-STARTPTS")
            if clip.speed != 1.0:
                steps.append(f"setpts=(PTS)/{clip.speed:.6f}")
        else:  # image/color already cut to clip.duration via input -t
            steps.append("setpts=PTS-STARTPTS")

        tr = clip.transform
        if tr.crop_w and tr.crop_h:
            steps.append(f"crop={tr.crop_w}:{tr.crop_h}:{tr.crop_x}:{tr.crop_y}")

        # fit inside canvas preserving aspect, pad to exact WxH
        target_w = max(2, int(round(W * tr.scale)))
        target_h = max(2, int(round(H * tr.scale)))
        steps.append(
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease")
        steps.append(f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={self.p.globals.background}")
        steps.append("setsar=1")
        steps.append(f"fps={FPS}")
        steps.append("format=yuv420p")

        dur = clip.duration
        if clip.transition_in > 0:
            steps.append(f"fade=t=in:st=0:d={clip.transition_in:.6f}")
        if clip.transition_out > 0:
            st = max(0.0, dur - clip.transition_out)
            steps.append(f"fade=t=out:st={st:.6f}:d={clip.transition_out:.6f}")

        # offset onto the timeline
        steps.append(f"setpts=PTS+{clip.timeline_in:.6f}/TB")
        return f"{src}{','.join(steps)}[{out_label}]"

    # ------------------------------------------------------------------ #
    def _audio_chain(self, idx: int, clip: Clip, asset: Asset, out_label: str) -> str:
        steps: list[str] = []
        src = f"[{idx}:a]"
        steps.append(f"atrim=start={clip.src_in:.6f}:end={clip.src_out:.6f}")
        steps.append("asetpts=PTS-STARTPTS")
        if clip.speed != 1.0:
            # atempo supports 0.5..100 per stage; chain for extreme speeds
            s = clip.speed
            factors = []
            while s > 2.0:
                factors.append(2.0); s /= 2.0
            while s < 0.5:
                factors.append(0.5); s /= 0.5
            factors.append(s)
            steps += [f"atempo={f:.6f}" for f in factors]
        if clip.volume != 1.0:
            steps.append(f"volume={clip.volume:.6f}")
        dur = clip.duration
        if clip.transition_in > 0:
            steps.append(f"afade=t=in:st=0:d={clip.transition_in:.6f}")
        if clip.transition_out > 0:
            st = max(0.0, dur - clip.transition_out)
            steps.append(f"afade=t=out:st={st:.6f}:d={clip.transition_out:.6f}")
        steps.append("aresample=async=1")
        delay_ms = int(round(clip.timeline_in * 1000))
        steps.append(f"adelay=delays={delay_ms}:all=1")
        return f"{src}{','.join(steps)}[{out_label}]"

    # ------------------------------------------------------------------ #
    def _drawtext(self, layer: TextLayer) -> str:
        parts = [
            f"text='{_esc_drawtext(layer.text)}'",
            f"fontsize={layer.font_size}",
            f"fontcolor={layer.font_color}",
            f"x={layer.x}", f"y={layer.y}",
            f"enable='between(t,{layer.timeline_in:.6f},{layer.timeline_out:.6f})'",
        ]
        if layer.box:
            parts += ["box=1", f"boxcolor={layer.box_color}", "boxborderw=12"]
        return "drawtext=" + ":".join(parts)

    # ------------------------------------------------------------------ #
    def build(self) -> CompiledRender:
        p = self.p
        duration = p.duration
        if duration <= 0:
            raise ValueError("project has zero duration; add at least one clip")

        inputs: list[list[str]] = []
        # input 0 = base canvas
        inputs.append(["-f", "lavfi", "-t", f"{duration:.6f}",
                       "-i", f"color=c={p.globals.background}:s={p.width}x{p.height}:"
                             f"r={p.fps}"])

        video_chains: list[str] = []
        audio_chains: list[str] = []
        video_labels: list[tuple[float, str]] = []  # (timeline_in, label) per layer
        audio_labels: list[str] = []

        clip_idx = 0
        for track in p.tracks:
            for clip in track.clips:
                asset = p.asset(clip.asset)
                inputs.append(self._input_args(clip, asset, track.kind))
                in_index = len(inputs) - 1

                wants_video = track.kind in (TrackKind.video, TrackKind.caption) and \
                    asset.type in (AssetType.video, AssetType.image, AssetType.color)
                wants_audio = asset.type in (AssetType.video, AssetType.audio) and \
                    track.kind in (TrackKind.video, TrackKind.audio)

                if wants_video:
                    label = f"v{clip_idx}"
                    video_chains.append(self._video_chain(in_index, clip, asset, label))
                    video_labels.append((clip.timeline_in, label, clip))
                if wants_audio:
                    label = f"a{clip_idx}"
                    audio_chains.append(self._audio_chain(in_index, clip, asset, label))
                    audio_labels.append(label)
                clip_idx += 1

        graph: list[str] = []
        graph.extend(video_chains)
        graph.extend(audio_chains)

        # ---- composite video onto canvas (stable order: input order) ----- #
        prev = "0:v"
        # canvas needs fps/format normalization
        graph.append(f"[0:v]fps={p.fps},format=yuv420p[canvas]")
        prev = "canvas"
        for i, (_, label, clip) in enumerate(video_labels):
            out = f"comp{i}"
            tr = clip.transform
            x = f"(main_w-overlay_w)/2+{tr.pos_x}"
            y = f"(main_h-overlay_h)/2+{tr.pos_y}"
            graph.append(
                f"[{prev}][{label}]overlay=x={x}:y={y}:"
                f"enable='between(t,{clip.timeline_in:.6f},{clip.timeline_out:.6f})':"
                f"eof_action=pass[{out}]")
            prev = out

        # ---- burned-in text on top ---------------------------------------- #
        texts = [t for track in p.tracks for t in track.texts]
        if texts:
            chain = ",".join(self._drawtext(t) for t in texts)
            graph.append(f"[{prev}]{chain}[vout]")
        else:
            graph.append(f"[{prev}]null[vout]")

        maps = ["[vout]"]
        has_audio = bool(audio_labels)
        if has_audio:
            if len(audio_labels) == 1:
                graph.append(f"[{audio_labels[0]}]aresample={p.render_settings.audio_rate},"
                             f"loudnorm=I={p.globals.loudness_target}:LRA=11:TP=-1.5[aout]")
            else:
                ins = "".join(f"[{l}]" for l in audio_labels)
                graph.append(
                    f"{ins}amix=inputs={len(audio_labels)}:normalize=0:"
                    f"dropout_transition=0[amixed]")
                graph.append(
                    f"[amixed]aresample={p.render_settings.audio_rate},"
                    f"loudnorm=I={p.globals.loudness_target}:LRA=11:TP=-1.5[aout]")
            maps.append("[aout]")

        return CompiledRender(inputs=inputs, filter_complex=";".join(graph),
                              maps=maps, has_audio=has_audio, duration=duration,
                              fps=p.fps)

    # ------------------------------------------------------------------ #
    def render(self, out_path: str, *, dry_run: bool = False,
               quiet: bool = True) -> dict:
        compiled = self.build()
        rs = self.p.render_settings
        cmd = compiled.command(out_path, rs)
        if dry_run:
            return {"ok": True, "dry_run": True, "command": " ".join(shlex.quote(c) for c in cmd),
                    "filter_complex": compiled.filter_complex,
                    "duration": compiled.duration, "has_audio": compiled.has_audio,
                    "content_hash": self.p.content_hash()}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}):\n{tail}")
        return {"ok": True, "output": out_path, "duration": compiled.duration,
                "has_audio": compiled.has_audio, "content_hash": self.p.content_hash(),
                "size_bytes": os.path.getsize(out_path)}
