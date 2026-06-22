"""Embedding generation for semantic video search.

Generates multimodal embeddings for keyframes and audio segments
to enable similarity-based search across the media index.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Optional

from ..index.models import AssetIndex

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and manage embeddings for media content.

    Supports multiple providers: CLIP (local), OpenAI embeddings,
    or mock (hash-based) fallback.
    """

    def __init__(self, provider: str = "clip", dim: int = 512,
                 api_key: Optional[str] = None):
        self.provider = provider
        self.dim = dim
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY", "")

    def index_asset(self, uri: str, index: AssetIndex) -> list[str]:
        """Generate embeddings for keyframes in an asset index."""
        refs = []
        for scene in index.scenes:
            for shot in scene.shots:
                embedding = self._embed_keyframe(uri, shot.start)
                if embedding is not None:
                    ref = f"{index.asset_id}_{shot.id}"
                    refs.append(ref)
        return refs

    def embed_text(self, text: str) -> Optional[list[float]]:
        """Generate an embedding for a text query."""
        if self.provider == "openai":
            return self._embed_text_openai(text)
        return self._embed_text_mock(text)

    def embed_image(self, image_path: str) -> Optional[list[float]]:
        """Generate an embedding for an image."""
        if self.provider == "clip":
            return self._embed_image_clip(image_path)
        return None

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _embed_keyframe(self, uri: str, timestamp: float) -> Optional[list[float]]:
        """Extract a keyframe and generate its embedding."""
        keyframe = self._extract_keyframe(uri, timestamp)
        if keyframe is None:
            return None
        try:
            embedding = self.embed_image(keyframe)
            return embedding
        finally:
            try:
                os.unlink(keyframe)
            except OSError:
                pass

    def _embed_image_clip(self, image_path: str) -> Optional[list[float]]:
        """Use local CLIP model for image embedding."""
        try:
            import torch
            from PIL import Image
            model, preprocess = self._load_clip()
            if model is None:
                return None
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                embedding = model.encode_image(image_input)
            return embedding[0].tolist()
        except ImportError:
            logger.warning("CLIP not available, using mock embeddings")
            return self._mock_embedding()

    def _embed_text_openai(self, text: str) -> Optional[list[float]]:
        """Use OpenAI text-embedding-3-small for text embedding."""
        if not self.api_key:
            return None
        import urllib.request
        payload = {"model": "text-embedding-3-small", "input": text}
        url = "https://api.openai.com/v1/embeddings"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data["data"][0]["embedding"]

    def _embed_text_mock(self, text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        values = [b / 255.0 for b in h]
        while len(values) < self.dim:
            values.extend(values[:self.dim - len(values)])
        return values[:self.dim]

    def _mock_embedding(self) -> list[float]:
        import random
        random.seed(42)
        return [random.gauss(0, 1) for _ in range(self.dim)]

    def _load_clip(self):
        try:
            import torch
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            return model, preprocess
        except (ImportError, Exception):
            return None, None

    def _extract_keyframe(self, uri: str, timestamp: float) -> Optional[str]:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-nostdin",
                "-ss", f"{timestamp:.3f}", "-i", uri,
                "-frames:v", "1", "-q:v", "2", tmp.name,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                   encoding="utf-8", errors="replace")
            if proc.returncode == 0 and os.path.getsize(tmp.name) > 0:
                return tmp.name
            os.unlink(tmp.name)
        except Exception:
            pass
        return None
