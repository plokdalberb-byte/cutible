"""Deterministic QC engine — the agent's cheap, exact "eyes" (plan §3.2).

No LLM. Probes a rendered file with ffprobe/ffmpeg and returns a structured
list of violations with timecodes, so the QC/Reviewer agent can decide whether
to gate release or loop back and fix. This is the half of the perception loop
that is fast and precise; the VLM review (semantic critique) is the other half.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Violation:
    code: str
    severity: str           # "error" | "warn" | "info"
    message: str
    at: Optional[float] = None

    def to_dict(self) -> dict:
        d = {"code": self.code, "severity": self.severity, "message": self.message}
        if self.at is not None:
            d["at"] = self.at
        return d


@dataclass
class QCReport:
    path: str
    passed: bool
    duration: float
    has_video: bool
    has_audio: bool
    integrated_lufs: Optional[float] = None
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "passed": self.passed,
            "duration": self.duration,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "integrated_lufs": self.integrated_lufs,
            "violations": [v.to_dict() for v in self.violations],
        }


def _ffprobe(path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", path]
    out = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {out.stderr[-400:]}")
    return json.loads(out.stdout)


def run_qc(path: str, *, expected_duration: Optional[float] = None,
           loudness_target: float = -14.0, loudness_tol: float = 2.0,
           duration_tol: float = 0.3, detect_black: bool = True) -> QCReport:
    """Probe a rendered video and return a structured QC report."""
    info = _ffprobe(path)
    streams = info.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    duration = float(info.get("format", {}).get("duration", 0.0) or 0.0)

    report = QCReport(path=path, passed=True, duration=round(duration, 3),
                      has_video=bool(vstreams), has_audio=bool(astreams))
    V = report.violations

    if not vstreams:
        V.append(Violation("no_video", "error", "rendered file has no video stream"))
    if expected_duration is not None:
        delta = abs(duration - expected_duration)
        if delta > duration_tol:
            V.append(Violation(
                "duration_mismatch", "error",
                f"duration {duration:.2f}s deviates from expected "
                f"{expected_duration:.2f}s by {delta:.2f}s (tol {duration_tol}s)"))

    # ---- black frame detection (cheap, deterministic) ------------------ #
    if detect_black and vstreams:
        bd = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-vf", "blackdetect=d=0.3:pic_th=0.98", "-an", "-f", "null", "-"],
            capture_output=True, text=True)
        for m in re.finditer(r"black_start:(\d+\.?\d*)", bd.stderr):
            V.append(Violation("black_frames", "warn",
                               "sustained black frames detected", at=float(m.group(1))))

    # ---- loudness probe (EBU R128) ------------------------------------- #
    if astreams:
        ln = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-af", "ebur128=framelog=verbose", "-f", "null", "-"],
            capture_output=True, text=True)
        m = re.findall(r"I:\s*(-?\d+\.?\d*)\s*LUFS", ln.stderr)
        if m:
            lufs = float(m[-1])
            report.integrated_lufs = lufs
            if abs(lufs - loudness_target) > loudness_tol:
                V.append(Violation(
                    "loudness_off_target", "warn",
                    f"integrated loudness {lufs:.1f} LUFS vs target "
                    f"{loudness_target:.1f} (tol {loudness_tol})"))

    report.passed = not any(v.severity == "error" for v in V)
    return report
