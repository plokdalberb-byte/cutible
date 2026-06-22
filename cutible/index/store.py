"""Index storage and persistence.

Stores AssetIndex and NarrativeIndex objects, supports saving/loading
from JSON and embedding vector persistence for similarity search.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

from .models import AssetIndex, NarrativeIndex


class IndexStore:
    """Manages persistence and retrieval of semantic media indices."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self._indices: dict[str, AssetIndex] = {}
        self._narrative: Optional[NarrativeIndex] = None
        self._embeddings: dict[str, list[float]] = {}
        self._load_embeddings()

    def store_asset_index(self, index: AssetIndex) -> None:
        self._indices[index.asset_id] = index
        path = self._asset_path(index.asset_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(index.model_dump_json(indent=2))

    def load_asset_index(self, asset_id: str) -> Optional[AssetIndex]:
        if asset_id in self._indices:
            return self._indices[asset_id]
        path = self._asset_path(asset_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                index = AssetIndex.model_validate_json(f.read())
            self._indices[asset_id] = index
            return index
        return None

    def store_narrative(self, narrative: NarrativeIndex) -> None:
        self._narrative = narrative
        path = os.path.join(self.base_dir, "narrative.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(narrative.model_dump_json(indent=2))

    def load_narrative(self) -> Optional[NarrativeIndex]:
        if self._narrative is not None:
            return self._narrative
        path = os.path.join(self.base_dir, "narrative.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._narrative = NarrativeIndex.model_validate_json(f.read())
            return self._narrative
        return None

    def list_assets(self) -> list[str]:
        indices_dir = os.path.join(self.base_dir, "assets")
        if not os.path.exists(indices_dir):
            return []
        return [
            d.replace(".json", "")
            for d in os.listdir(indices_dir)
            if d.endswith(".json")
        ]

    def delete_asset(self, asset_id: str) -> bool:
        self._indices.pop(asset_id, None)
        path = self._asset_path(asset_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def _asset_path(self, asset_id: str) -> str:
        return os.path.join(self.base_dir, "assets", f"{asset_id}.json")

    def get_all_indices(self) -> list[AssetIndex]:
        result = []
        for asset_id in self.list_assets():
            idx = self.load_asset_index(asset_id)
            if idx is not None:
                result.append(idx)
        return result

    def build_narrative(self, project_id: str) -> NarrativeIndex:
        """Build a NarrativeIndex from all stored asset indices."""
        indices = self.get_all_indices()
        total_dur = sum(i.duration for i in indices)
        all_speakers = {}
        for ai in indices:
            for sp in ai.speakers:
                if sp.speaker_id not in all_speakers:
                    all_speakers[sp.speaker_id] = sp
        narrative = NarrativeIndex(
            project_id=project_id,
            total_duration=total_dur,
            speakers=list(all_speakers.values()),
            asset_indices=indices,
        )
        self.store_narrative(narrative)
        return narrative

    # ---- Embedding persistence ----

    def _embeddings_path(self) -> str:
        return os.path.join(self.base_dir, "embeddings.json")

    def _load_embeddings(self) -> None:
        path = self._embeddings_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self._embeddings = json.load(f)

    def _save_embeddings(self) -> None:
        path = self._embeddings_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._embeddings, f)

    def store_embedding(self, ref: str, vector: list[float]) -> None:
        """Store an embedding vector by reference string."""
        self._embeddings[ref] = vector
        self._save_embeddings()

    def load_embedding(self, ref: str) -> Optional[list[float]]:
        """Load an embedding vector by reference string."""
        return self._embeddings.get(ref)

    def get_all_embeddings(self) -> dict[str, list[float]]:
        """Return all stored embeddings."""
        return dict(self._embeddings)

    def search_semantic(self, query_vector: list[float],
                        top_k: int = 10,
                        min_score: float = 0.3) -> list[dict]:
        """Find the most similar embeddings using cosine similarity.

        Returns a list of {ref, score, vector} dicts sorted by score desc.
        """
        if not self._embeddings or not query_vector:
            return []

        results = []
        q_norm = math.sqrt(sum(x * x for x in query_vector))
        if q_norm == 0:
            return []

        for ref, vec in self._embeddings.items():
            if len(vec) != len(query_vector):
                continue
            dot = sum(a * b for a, b in zip(query_vector, vec))
            v_norm = math.sqrt(sum(x * x for x in vec))
            if v_norm == 0:
                continue
            score = dot / (q_norm * v_norm)
            if score >= min_score:
                results.append({"ref": ref, "score": score})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]
