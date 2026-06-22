"""Speaker diarization — assign speaker labels to transcript segments.

Supports multiple backends:
1. pyannote.audio (real diarization, requires API key + GPU recommended)
2. Energy-based heuristic (fallback, no dependencies)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from ..index.models import SpeakerProfile, TranscriptSegment

logger = logging.getLogger(__name__)


class Diarizer:
    """Speaker diarization with pyannote fallback to energy heuristic."""

    def __init__(self, provider: str = "pyannote",
                 pyannote_token: Optional[str] = None):
        self.provider = provider
        self.pyannote_token = pyannote_token or os.environ.get(
            "PYANNOTE_TOKEN", os.environ.get("HF_TOKEN", "")
        )
        self._pipeline = None

    def diarize(self, audio_path: str,
                segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Assign speaker labels to existing transcript segments.

        Tries pyannote first, falls back to energy heuristic.
        Returns segments with updated ``speaker`` fields.
        """
        if self.provider == "pyannote" and self.pyannote_token:
            try:
                return self._diarize_pyannote(audio_path, segments)
            except Exception as e:
                logger.warning(f"pyannote diarization failed: {e}, using heuristic")

        return self._diarize_energy_heuristic(segments)

    def _diarize_pyannote(self, audio_path: str,
                           segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Real diarization using pyannote.audio Pipeline."""
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio required: pip install 'cutible[diarization]'"
            )

        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.pyannote_token,
            )

        diarization = self._pipeline(audio_path)

        # Build a timeline of speaker assignments
        speaker_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append((turn.start, turn.end, speaker))

        # Assign speakers to segments based on overlap
        for seg in segments:
            seg.speaker = self._find_dominant_speaker(
                seg.start, seg.end, speaker_timeline
            )

        return segments

    def _find_dominant_speaker(self, start: float, end: float,
                                timeline: list[tuple[float, float, str]]) -> str:
        """Find the speaker with the most overlap for a time range."""
        speaker_times: dict[str, float] = {}
        for t_start, t_end, speaker in timeline:
            overlap_start = max(start, t_start)
            overlap_end = min(end, t_end)
            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start
                speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap

        if not speaker_times:
            return "unknown"
        return max(speaker_times, key=speaker_times.get)

    def _diarize_energy_heuristic(self,
                                   segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        """Energy-based heuristic: group consecutive segments by pause gaps.

        When there's a significant gap (>0.8s) between segments,
        assume a speaker change. This is better than the old
        midpoint-based alternation.
        """
        if not segments:
            return segments

        current_speaker = "speaker_0"
        speaker_idx = 0
        gap_threshold = 0.8

        for i, seg in enumerate(segments):
            if i > 0:
                gap = seg.start - segments[i - 1].end
                if gap >= gap_threshold:
                    speaker_idx += 1
                    current_speaker = f"speaker_{speaker_idx}"
            seg.speaker = current_speaker

        return segments


def build_speaker_profiles(segments: list[TranscriptSegment]) -> list[SpeakerProfile]:
    """Build SpeakerProfile objects from labeled transcript segments."""
    speakers: dict[str, SpeakerProfile] = {}
    for seg in segments:
        sp = seg.speaker
        if sp not in speakers:
            speakers[sp] = SpeakerProfile(
                speaker_id=sp,
                first_appearance=seg.start,
            )
        profile = speakers[sp]
        profile.total_speaking_time += seg.duration
        profile.segment_count += 1
        profile.last_appearance = max(profile.last_appearance, seg.end)
    return list(speakers.values())
