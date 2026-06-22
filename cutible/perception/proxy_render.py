"""Proxy renderer for fast preview generation.

Renders low-resolution proxy videos for the perception loop,
allowing the agent to quickly review its work without waiting
for full-quality renders.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from ..schema import Project

logger = logging.getLogger(__name__)


@dataclass
class ProxyConfig:
    """Configuration for proxy rendering."""

    width: int = 480
    height: int = 270
    fps: int = 15
    crf: int = 28
    audio_bitrate: str = "64k"
    max_duration: float = 300.0  # cap at 5 minutes for proxy

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


class ProxyRenderer:
    """Render fast low-resolution proxies for perception loop review.

    Proxies are ~10-20x faster to render than full quality, enabling
    the agent to iterate quickly through the perception loop.
    """

    def __init__(self, config: Optional[ProxyConfig] = None):
        self.config = config or ProxyConfig()

    def render_proxy(self, project: Project, output_path: str,
                     max_duration: Optional[float] = None) -> dict:
        """Render a low-resolution proxy of the project."""
        from ..compiler import FFmpegCompiler

        compiler = FFmpegCompiler(project)
        compiled = compiler.build()

        duration = min(compiled.duration, max_duration or self.config.max_duration)

        cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin"]
        for inp in compiled.inputs:
            cmd += inp
        cmd += ["-filter_complex", compiled.filter_complex]
        for m in compiled.maps:
            cmd += ["-map", m]
        cmd += [
            "-r", str(self.config.fps),
            "-vf", f"scale={self.config.resolution}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", str(self.config.crf),
            "-pix_fmt", "yuv420p",
        ]
        if compiled.has_audio:
            cmd += [
                "-c:a", "aac", "-b:a", self.config.audio_bitrate,
                "-ar", "44100",
            ]
        cmd += [
            "-t", f"{duration:.6f}",
            "-map_metadata", "-1",
            output_path,
        ]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                               encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            raise RuntimeError(f"Proxy render failed: {tail}")

        return {
            "ok": True,
            "output": output_path,
            "duration": duration,
            "resolution": self.config.resolution,
            "fps": self.config.fps,
            "size_bytes": os.path.getsize(output_path),
        }

    def render_clip_proxy(self, project: Project, clip_id: str,
                          output_path: str) -> dict:
        """Render a proxy of a single clip for isolated review."""
        from ..compiler import FFmpegCompiler

        clip = None
        for track in project.tracks:
            for c in track.clips:
                if c.id == clip_id:
                    clip = c
                    break
        if clip is None:
            raise ValueError(f"clip {clip_id!r} not found")

        asset = project.asset(clip.asset)
        if not asset.uri:
            raise ValueError(f"clip {clip_id!r} has no media URI")

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostdin",
            "-ss", f"{clip.src_in:.3f}",
            "-i", asset.uri,
            "-t", f"{clip.duration:.3f}",
            "-vf", f"scale={self.config.resolution}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", str(self.config.crf),
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                               encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"Clip proxy render failed")

        return {
            "ok": True,
            "clip_id": clip_id,
            "output": output_path,
            "duration": clip.duration,
        }

    def render_segment(self, project: Project, start: float, end: float,
                       output_path: str) -> dict:
        """Render a segment of the timeline as proxy."""
        from ..compiler import FFmpegCompiler

        compiler = FFmpegCompiler(project)
        compiled = compiler.build()

        cmd = ["ffmpeg", "-y", "-hide_banner", "-nostdin"]
        for inp in compiled.inputs:
            cmd += inp
        cmd += ["-filter_complex", compiled.filter_complex]
        for m in compiled.maps:
            cmd += ["-map", m]
        cmd += [
            "-ss", f"{start:.3f}",
            "-t", f"{end - start:.3f}",
            "-r", str(self.config.fps),
            "-vf", f"scale={self.config.resolution}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", str(self.config.crf),
            "-pix_fmt", "yuv420p",
        ]
        if compiled.has_audio:
            cmd += ["-c:a", "aac", "-b:a", self.config.audio_bitrate, "-ar", "44100"]
        cmd += ["-map_metadata", "-1", output_path]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                               encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(f"Segment proxy render failed")

        return {
            "ok": True,
            "output": output_path,
            "start": start,
            "end": end,
            "duration": end - start,
        }
