"""Scene detection using PySceneDetect or ffmpeg-based detection.

Detects shot boundaries and groups them into scenes.
"""

from __future__ import annotations

import logging
import subprocess
import re
from typing import Optional

from ..index.models import Scene, Shot, VisualDescription

logger = logging.getLogger(__name__)


class SceneDetector:
    """Detects scene and shot boundaries in video files.

    Uses ffmpeg's scene detection filter as the primary method,
    with PySceneDetect as an optional enhancement.
    """

    def __init__(self, threshold: float = 0.3, min_scene_len: float = 0.5):
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect(self, uri: str, asset_id: str = "") -> list[Scene]:
        """Detect scenes in a video file."""
        shots = self._detect_shots_ffmpeg(uri, asset_id)
        scenes = self._group_into_scenes(shots, asset_id)
        logger.info(f"  Detected {len(scenes)} scenes, {len(shots)} shots")
        return scenes

    def _detect_shots_ffmpeg(self, uri: str, asset_id: str) -> list[Shot]:
        """Use ffmpeg's scene detection filter to find shot boundaries."""
        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin",
            "-i", uri,
            "-vf", f"select='gt(scene,{self.threshold})',showinfo",
            "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                   encoding="utf-8", errors="replace")
            timestamps = [0.0]
            for line in proc.stderr.splitlines():
                m = re.search(r"pts_time:(\d+\.?\d*)", line)
                if m:
                    t = float(m.group(1))
                    if t - timestamps[-1] >= self.min_scene_len:
                        timestamps.append(t)
            duration = self._get_duration(uri)
            timestamps.append(duration)
            shots = []
            for i in range(len(timestamps) - 1):
                shots.append(Shot(
                    id=f"{asset_id}_shot_{i}",
                    asset_id=asset_id,
                    start=timestamps[i],
                    end=timestamps[i + 1],
                ))
            return shots
        except Exception as e:
            logger.warning(f"  Scene detection failed: {e}, creating single shot")
            duration = self._get_duration(uri)
            return [Shot(id=f"{asset_id}_shot_0", asset_id=asset_id,
                         start=0.0, end=duration)]

    def _group_into_scenes(self, shots: list[Shot], asset_id: str) -> list[Scene]:
        """Group shots into semantic scenes using temporal proximity."""
        if not shots:
            return []
        scenes: list[Scene] = []
        current_shots = [shots[0]]
        for shot in shots[1:]:
            gap = shot.start - current_shots[-1].end
            if gap > 5.0 or (len(current_shots) >= 10):
                scenes.append(Scene(
                    id=f"{asset_id}_scene_{len(scenes)}",
                    asset_id=asset_id,
                    start=current_shots[0].start,
                    end=current_shots[-1].end,
                    shots=current_shots,
                ))
                current_shots = [shot]
            else:
                current_shots.append(shot)
        scenes.append(Scene(
            id=f"{asset_id}_scene_{len(scenes)}",
            asset_id=asset_id,
            start=current_shots[0].start,
            end=current_shots[-1].end,
            shots=current_shots,
        ))
        return scenes

    def _get_duration(self, uri: str) -> float:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
            "-of", "csv=p=0", uri,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                   encoding="utf-8", errors="replace")
            return float(proc.stdout.strip())
        except Exception:
            return 0.0
