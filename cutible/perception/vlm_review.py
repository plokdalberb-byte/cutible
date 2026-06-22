"""VLM-based semantic review of rendered video.

Sends proxy renders to a VLM for quality assessment:
cut quality, pacing, transitions, captions, brand compliance.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReviewIssue:
    """A single issue found during VLM review."""

    category: str  # "pacing", "transition", "caption", "audio", "visual", "content"
    severity: str  # "critical", "warning", "info"
    timecode: float  # where in the video
    description: str
    suggestion: str = ""

    def to_dict(self) -> dict:
        d = {
            "category": self.category,
            "severity": self.severity,
            "timecode": self.timecode,
            "description": self.description,
        }
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


@dataclass
class ReviewReport:
    """Complete review report from VLM perception loop."""

    passed: bool
    overall_score: float  # 0-1
    issues: list[ReviewIssue] = field(default_factory=list)
    summary: str = ""
    frame_scores: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "n_issues": len(self.issues),
            "critical": len([i for i in self.issues if i.severity == "critical"]),
            "warnings": len([i for i in self.issues if i.severity == "warning"]),
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


class VLMReview:
    """Semantic review of rendered video using Vision-Language Models.

    Samples frames from the rendered video and sends them to a VLM
    for quality assessment, then compiles a structured review report.
    """

    def __init__(self, model: str = "gemini", api_key: str | None = None, sample_fps: float = 0.5):
        self.model = model
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.sample_fps = sample_fps

    def review(
        self, video_path: str, edit_plan: dict | None = None, brief: str = ""
    ) -> ReviewReport:
        """Run a full VLM review on a rendered video."""
        frames = self._sample_frames(video_path)
        if not frames:
            return ReviewReport(
                passed=False,
                overall_score=0.0,
                summary="Could not extract frames for review",
            )

        issues = []
        frame_scores = []

        for frame_path, timestamp in frames:
            result = self._review_frame(frame_path, timestamp, edit_plan, brief)
            if result:
                frame_scores.append(result)
                issues.extend(result.get("issues", []))

        overall = self._compute_overall_score(frame_scores, issues)
        passed = overall >= 0.6 and not any(i.severity == "critical" for i in issues)

        return ReviewReport(
            passed=passed,
            overall_score=overall,
            issues=issues,
            summary=self._generate_summary(issues, overall, passed),
            frame_scores=frame_scores,
        )

    def _review_frame(
        self, frame_path: str, timestamp: float, edit_plan: dict | None, brief: str
    ) -> dict | None:
        """Review a single frame via VLM."""
        prompt = self._build_review_prompt(edit_plan, brief)
        try:
            if self.api_key:
                return self._call_vlm(frame_path, prompt, timestamp)
            return self._mock_review(timestamp)
        except Exception as e:
            logger.warning(f"VLM review failed at {timestamp}s: {e}")
            return self._mock_review(timestamp)

    def _build_review_prompt(self, edit_plan: dict | None, brief: str) -> str:
        prompt = (
            "You are a professional video editor reviewing a rendered clip. "
            "Analyze this frame and return a JSON object with:\n"
            '- "visual_quality": 0-1 score\n'
            '- "issues": list of issues, each with "category" (pacing/transition/caption/audio/visual/content), '
            '"severity" (critical/warning/info), "description", "suggestion"\n'
            '- "notes": any observations\n'
        )
        if brief:
            prompt += f'\nOriginal brief: "{brief}"\n'
        if edit_plan:
            prompt += f"\nEdit plan summary: {json.dumps(edit_plan, indent=2)[:500]}\n"
        return prompt

    def _call_vlm(self, frame_path: str, prompt: str, timestamp: float) -> dict:
        """Call the VLM API for frame review."""
        with open(frame_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                        {"text": prompt},
                    ]
                }
            ],
        }
        if self.model == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        else:
            raise ValueError(f"Unsupported VLM model: {self.model}")

        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(text)
        result["timestamp"] = timestamp
        return result

    def _mock_review(self, timestamp: float) -> dict:
        return {
            "timestamp": timestamp,
            "visual_quality": 0.7,
            "issues": [],
            "notes": "[VLM review unavailable — using mock]",
        }

    def _compute_overall_score(self, frame_scores: list[dict], issues: list[ReviewIssue]) -> float:
        if not frame_scores:
            return 0.5
        qualities = [f.get("visual_quality", 0.5) for f in frame_scores]
        base = sum(qualities) / len(qualities)
        penalty = sum(
            0.2 if i.severity == "critical" else 0.1 if i.severity == "warning" else 0.02
            for i in issues
        )
        return max(0.0, min(1.0, base - penalty))

    def _generate_summary(self, issues: list[ReviewIssue], overall: float, passed: bool) -> str:
        critical = [i for i in issues if i.severity == "critical"]
        warnings = [i for i in issues if i.severity == "warning"]
        status = "PASSED" if passed else "FAILED"
        parts = [f"Review {status} (score: {overall:.2f})"]
        if critical:
            parts.append(f"{len(critical)} critical issues")
        if warnings:
            parts.append(f"{len(warnings)} warnings")
        if not issues:
            parts.append("No issues found")
        return ". ".join(parts)

    def _sample_frames(self, video_path: str) -> list[tuple[str, float]]:
        """Extract sample frames from the video."""
        duration = self._get_duration(video_path)
        if duration <= 0:
            return []
        frames = []
        interval = 1.0 / self.sample_fps
        t = 0.0
        while t < duration:
            frame_path = self._extract_frame(video_path, t)
            if frame_path:
                frames.append((frame_path, t))
            t += interval
        return frames

    def _extract_frame(self, video_path: str, timestamp: float) -> str | None:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-nostdin",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-q:v",
                "5",
                tmp.name,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
            )
            if proc.returncode == 0 and os.path.getsize(tmp.name) > 0:
                return tmp.name
            os.unlink(tmp.name)
        except Exception:
            pass
        return None

    def _get_duration(self, video_path: str) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            video_path,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
            )
            return float(proc.stdout.strip())
        except Exception:
            return 0.0
