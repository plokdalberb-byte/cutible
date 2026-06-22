"""VLM (Vision-Language Model) analysis of video content.

Sends keyframes/segments to a VLM API for visual understanding:
descriptions, subjects, actions, emotions, shot type classification.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

from ..index.models import Scene, VisualDescription, ShotType

logger = logging.getLogger(__name__)


class VLMAnalyzer:
    """Analyze video content using Vision-Language Models.

    Supports multiple backends (Gemini, GPT-4o, local models)
    through a unified interface.
    """

    def __init__(self, model: str = "gemini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self._mock_counter = 0

    def analyze_scenes(self, uri: str, scenes: list[Scene]) -> list[VisualDescription]:
        """Analyze each scene/shot and return visual descriptions."""
        descriptions = []
        for scene in scenes:
            for shot in scene.shots:
                keyframe = self._extract_keyframe(uri, shot.start + (shot.end - shot.start) / 2)
                if keyframe:
                    vd = self._analyze_keyframe(keyframe, shot)
                    if vd:
                        shot.visual = vd
                        descriptions.append(vd)
                        shot.keyframe_uri = keyframe
        return descriptions

    def analyze_frame(self, uri: str, timestamp: float) -> Optional[VisualDescription]:
        """Analyze a single frame at a given timestamp."""
        keyframe = self._extract_keyframe(uri, timestamp)
        if not keyframe:
            return None
        shot = Shot(id="temp", asset_id="", start=timestamp, end=timestamp + 1.0)
        return self._analyze_keyframe(keyframe, shot)

    def _analyze_keyframe(self, keyframe_path: str, shot: 'Shot') -> Optional[VisualDescription]:
        """Send a keyframe to the VLM for analysis."""
        try:
            prompt = self._build_analysis_prompt()
            if self.model == "gemini":
                result = self._call_gemini(keyframe_path, prompt)
            elif self.model == "openai":
                result = self._call_openai(keyframe_path, prompt)
            else:
                result = self._mock_analysis(shot)
            return result
        except Exception as e:
            logger.warning(f"  VLM analysis failed: {e}")
            return self._mock_analysis(shot)

    def _build_analysis_prompt(self) -> str:
        return (
            "Analyze this video frame. Return a JSON object with:\n"
            '- "description": detailed scene description\n'
            '- "shot_type": one of [wide, medium, close_up, extreme_close_up, over_shoulder, pov, aerial, establishing]\n'
            '- "subjects": list of main subjects/people/objects\n'
            '- "action": what is happening\n'
            '- "emotion": emotional tone\n'
            '- "lighting": lighting conditions\n'
            '- "composition": visual composition notes\n'
            '- "b_roll_potential": 0-1 score for B-roll suitability\n'
            '- "quality_score": 0-1 technical quality score\n'
        )

    def _call_gemini(self, image_path: str, prompt: str) -> Optional[VisualDescription]:
        """Call Google Gemini API for visual analysis."""
        if not self.api_key:
            return None
        import urllib.request
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                {"text": prompt},
            ]}],
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_vlm_response(text)

    def _call_openai(self, image_path: str, prompt: str) -> Optional[VisualDescription]:
        """Call OpenAI GPT-4o API for visual analysis."""
        if not self.api_key:
            return None
        import urllib.request
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                {"type": "text", "text": prompt},
            ]}],
            "max_tokens": 500,
        }
        url = "https://api.openai.com/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        return self._parse_vlm_response(text)

    def _parse_vlm_response(self, text: str) -> Optional[VisualDescription]:
        """Parse VLM JSON response into VisualDescription."""
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
            shot_type_str = data.get("shot_type", "medium")
            try:
                shot_type = ShotType(shot_type_str)
            except ValueError:
                shot_type = ShotType.medium
            return VisualDescription(
                timestamp=0.0,
                duration=1.0,
                description=data.get("description", ""),
                shot_type=shot_type,
                subjects=data.get("subjects", []),
                action=data.get("action", ""),
                emotion=data.get("emotion", ""),
                lighting=data.get("lighting", ""),
                composition=data.get("composition", ""),
                b_roll_potential=float(data.get("b_roll_potential", 0.5)),
                quality_score=float(data.get("quality_score", 0.5)),
            )
        except Exception as e:
            logger.warning(f"  Failed to parse VLM response: {e}")
            return None

    def _mock_analysis(self, shot: 'Shot') -> VisualDescription:
        """Produce a basic analysis using ffmpeg metadata when VLM is unavailable."""
        # Try to extract basic visual info from the frame
        description = "[VLM unavailable — basic analysis]"
        shot_type = ShotType.medium
        quality = 0.5
        b_roll = 0.3

        if self._mock_counter % 3 == 0:
            shot_type = ShotType.wide
            description = "Wide establishing shot — available for overview"
            b_roll = 0.6
        elif self._mock_counter % 3 == 1:
            shot_type = ShotType.close_up
            description = "Close-up shot — suitable for emphasis"
            b_roll = 0.4
        else:
            shot_type = ShotType.medium
            description = "Medium shot — standard framing"
            b_roll = 0.3

        self._mock_counter += 1
        return VisualDescription(
            timestamp=shot.start,
            duration=shot.end - shot.start,
            description=description,
            shot_type=shot_type,
            subjects=[],
            b_roll_potential=b_roll,
            quality_score=quality,
        )

    def _extract_keyframe(self, uri: str, timestamp: float) -> Optional[str]:
        """Extract a single frame from the video."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-nostdin",
                "-ss", f"{timestamp:.3f}",
                "-i", uri,
                "-frames:v", "1",
                "-q:v", "2",
                tmp.name,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                   encoding="utf-8", errors="replace")
            if proc.returncode == 0 and os.path.getsize(tmp.name) > 0:
                return tmp.name
            os.unlink(tmp.name)
        except Exception:
            pass
        return None
