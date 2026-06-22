"""Ingest pipeline orchestrator.

Coordinates all ingest stages: decode → scene detection → transcription →
VLM analysis → audio analysis → embeddings → narrative index build.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Protocol

from ..index.models import AssetIndex, NarrativeIndex
from ..index.store import IndexStore
from .scenes import SceneDetector
from .audio_transcribe import AudioTranscriber
from .vlm import VLMAnalyzer
from .audio_analysis import AudioAnalyzer
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class IngestConfig:
    """Configuration for the ingest pipeline."""

    proxy_width: int = 480
    proxy_fps: int = 1
    whisper_model: str = "base"
    whisper_language: Optional[str] = None
    vlm_model: str = "gemini"
    vlm_api_key: Optional[str] = None
    embedding_provider: str = "clip"
    embedding_dim: int = 512
    index_dir: str = ".cutible/index"
    skip_vlm: bool = False
    skip_embeddings: bool = False
    skip_proxy: bool = False


@dataclass
class IngestResult:
    """Result of ingesting a single asset."""

    asset_id: str
    uri: str
    index: Optional[AssetIndex] = None
    error: Optional[str] = None
    proxy_path: Optional[str] = None
    keyframes_dir: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.index is not None and self.error is None

    def to_dict(self) -> dict:
        d = {
            "asset_id": self.asset_id,
            "uri": self.uri,
            "success": self.success,
            "proxy_path": self.proxy_path,
        }
        if self.error:
            d["error"] = self.error
        if self.index:
            d["index_summary"] = self.index.summary()
        return d


class IngestPipeline:
    """Orchestrates the full ingest pipeline for one or more assets.

    Usage::

        pipeline = IngestPipeline(config)
        result = pipeline.ingest_asset("speaker", "/path/to/video.mp4")
        narrative = pipeline.build_narrative("project_1")
    """

    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()
        self.store = IndexStore(self.config.index_dir)
        self.scene_detector = SceneDetector()
        self.transcriber = AudioTranscriber(
            model_name=self.config.whisper_model,
            language=self.config.whisper_language,
        )
        self.vlm = VLMAnalyzer(
            model=self.config.vlm_model,
            api_key=self.config.vlm_api_key,
        )
        self.audio_analyzer = AudioAnalyzer()
        self.embedder = EmbeddingGenerator(
            provider=self.config.embedding_provider,
            dim=self.config.embedding_dim,
        )

    def ingest_asset(self, asset_id: str, uri: str,
                     duration_hint: Optional[float] = None) -> IngestResult:
        """Run the full ingest pipeline on a single media file."""
        logger.info(f"Starting ingest for {asset_id}: {uri}")
        result = IngestResult(asset_id=asset_id, uri=uri)

        if not os.path.exists(uri):
            result.error = f"File not found: {uri}"
            return result

        try:
            # Stage 1: extract basic metadata
            metadata = self._extract_metadata(uri)
            duration = duration_hint or metadata.get("duration", 0.0)
            fps = int(metadata.get("fps", 30))
            width = int(metadata.get("width", 1920))
            height = int(metadata.get("height", 1080))

            index = AssetIndex(
                asset_id=asset_id,
                uri=uri,
                duration=duration,
                fps=fps,
                width=width,
                height=height,
            )

            # Stage 2: scene detection
            logger.info(f"  Detecting scenes...")
            scenes = self.scene_detector.detect(uri, asset_id)
            index.scenes = scenes

            # Stage 3: transcription
            logger.info(f"  Transcribing audio...")
            transcript = self.transcriber.transcribe(uri)
            index.transcript = transcript

            # Stage 4: VLM analysis (optional)
            if not self.config.skip_vlm:
                logger.info(f"  Running VLM analysis...")
                visual_descs = self.vlm.analyze_scenes(uri, scenes)
                index.visual_descriptions = visual_descs
                for scene in index.scenes:
                    for shot in scene.shots:
                        for vd in visual_descs:
                            if vd.timestamp >= shot.start and vd.timestamp < shot.end:
                                shot.visual = vd

            # Stage 5: audio analysis
            logger.info(f"  Analyzing audio...")
            audio_features = self.audio_analyzer.analyze(uri)
            index.audio_features = audio_features
            beats = self.audio_analyzer.detect_beats(uri)
            index.beat_times = beats
            tempo = self.audio_analyzer.estimate_tempo(uri)
            index.tempo_bpm = tempo
            silences = self.audio_analyzer.detect_silences(uri)
            index.silence_ranges = silences

            # Stage 6: speaker diarization (via transcriber)
            logger.info(f"  Diarizing speakers...")
            speakers = self.transcriber.diarize(uri)
            index.speakers = speakers

            # Stage 7: embeddings (optional)
            if not self.config.skip_embeddings:
                logger.info(f"  Generating embeddings...")
                embedding_refs = self.embedder.index_asset(uri, index)
                index.embedding_refs = embedding_refs

            # Store the completed index
            self.store.store_asset_index(index)
            result.index = index
            logger.info(f"  Ingest complete: {len(index.scenes)} scenes, "
                        f"{len(index.transcript)} transcript segments")

        except Exception as e:
            logger.error(f"  Ingest failed: {e}")
            result.error = str(e)

        return result

    def ingest_assets(self, assets: list[tuple[str, str]]) -> list[IngestResult]:
        """Ingest multiple assets sequentially."""
        results = []
        for asset_id, uri in assets:
            results.append(self.ingest_asset(asset_id, uri))
        return results

    def build_narrative(self, project_id: str) -> NarrativeIndex:
        """Build cross-asset narrative index from all ingested assets."""
        narrative = self.store.build_narrative(project_id)
        logger.info(f"Built narrative index: {narrative.total_duration:.1f}s "
                    f"across {len(narrative.asset_indices)} assets")
        return narrative

    def get_index(self, asset_id: str) -> Optional[AssetIndex]:
        return self.store.load_asset_index(asset_id)

    def get_narrative(self) -> Optional[NarrativeIndex]:
        return self.store.load_narrative()

    def _extract_metadata(self, uri: str) -> dict:
        """Extract media metadata using ffprobe."""
        import json
        import subprocess

        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", uri,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                   encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                return {}
            info = json.loads(proc.stdout)
            result = {}
            fmt = info.get("format", {})
            result["duration"] = float(fmt.get("duration", 0))
            for s in info.get("streams", []):
                if s.get("codec_type") == "video":
                    result["width"] = int(s.get("width", 1920))
                    result["height"] = int(s.get("height", 1080))
                    r_frame_rate = s.get("r_frame_rate", "30/1")
                    if "/" in r_frame_rate:
                        num, den = r_frame_rate.split("/")
                        result["fps"] = round(int(num) / int(den))
                    else:
                        result["fps"] = int(float(r_frame_rate))
                    break
            return result
        except Exception:
            return {}
