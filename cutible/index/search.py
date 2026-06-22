"""Search interface for the Semantic Media Index.

Provides text search, time-range queries, and vector similarity search
(using persisted embeddings).
"""

from __future__ import annotations

from .models import Shot
from .store import IndexStore


class IndexSearcher:
    """Unified search interface over all indexed content."""

    def __init__(self, store: IndexStore):
        self.store = store

    def search_text(self, query: str) -> list[dict]:
        """Full-text search across transcripts and visual descriptions."""
        results = []
        for idx in self.store.get_all_indices():
            q = query.lower()
            for seg in idx.transcript:
                if q in seg.text.lower():
                    results.append(
                        {
                            "asset_id": idx.asset_id,
                            "type": "transcript",
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "speaker": seg.speaker,
                        }
                    )
            for vd in idx.visual_descriptions:
                if q in vd.description.lower() or q in vd.action.lower():
                    results.append(
                        {
                            "asset_id": idx.asset_id,
                            "type": "visual",
                            "start": vd.timestamp,
                            "end": vd.timestamp + vd.duration,
                            "text": vd.description,
                        }
                    )
        return sorted(results, key=lambda r: r["start"])

    def search_semantic(self, query: str, top_k: int = 10) -> list[dict]:
        """Vector similarity search using persisted embeddings.

        Requires embeddings to have been generated and stored via
        EmbeddingGenerator + IndexStore.store_embedding().
        """
        from ..ingest.embeddings import EmbeddingGenerator

        embedder = EmbeddingGenerator(provider="mock")
        query_vector = embedder.embed_text(query)
        if query_vector is None:
            return []

        hits = self.store.search_semantic(query_vector, top_k=top_k, min_score=0.2)
        results = []
        for hit in hits:
            ref = hit["ref"]
            parts = ref.split("_", 1)
            asset_id = parts[0] if parts else ref
            shot_id = parts[1] if len(parts) > 1 else ""
            results.append(
                {
                    "ref": ref,
                    "asset_id": asset_id,
                    "shot_id": shot_id,
                    "score": hit["score"],
                    "type": "semantic",
                }
            )
        return results

    def search_time_range(
        self, start: float, end: float, asset_id: str | None = None
    ) -> list[Shot]:
        """Find all shots within a time range."""
        shots = []
        for idx in self.store.get_all_indices():
            if asset_id and idx.asset_id != asset_id:
                continue
            shots.extend(idx.shots_in_range(start, end))
        return shots

    def search_speaker(self, speaker_id: str) -> list[dict]:
        """Find all segments where a speaker is talking."""
        results = []
        for idx in self.store.get_all_indices():
            for seg in idx.transcript:
                if seg.speaker == speaker_id:
                    results.append(
                        {
                            "asset_id": idx.asset_id,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                        }
                    )
        return sorted(results, key=lambda r: r["start"])

    def search_silence(self, min_duration: float = 0.5) -> list[dict]:
        """Find silence ranges across all assets."""
        results = []
        for idx in self.store.get_all_indices():
            for start, end in idx.silence_ranges:
                dur = end - start
                if dur >= min_duration:
                    results.append(
                        {
                            "asset_id": idx.asset_id,
                            "start": start,
                            "end": end,
                            "duration": dur,
                        }
                    )
        return sorted(results, key=lambda r: r["start"])

    def search_b_roll(self, query: str, min_score: float = 0.5) -> list[dict]:
        """Find shots suitable as B-roll for a given topic."""
        results = []
        q = query.lower()
        for idx in self.store.get_all_indices():
            for vd in idx.visual_descriptions:
                if vd.b_roll_potential >= min_score:
                    matches_query = (
                        q in vd.description.lower()
                        or q in vd.action.lower()
                        or any(q in s.lower() for s in vd.subjects)
                    )
                    if matches_query or not query:
                        results.append(
                            {
                                "asset_id": idx.asset_id,
                                "start": vd.timestamp,
                                "end": vd.timestamp + vd.duration,
                                "description": vd.description,
                                "b_roll_score": vd.b_roll_potential,
                            }
                        )
        return sorted(results, key=lambda r: r["b_roll_score"], reverse=True)

    def find_best_segment(self, query: str, duration: float = 10.0) -> dict | None:
        """Find the best segment matching a query for a given duration."""
        results = self.search_text(query)
        if not results:
            results = self.search_b_roll(query)
        if not results:
            return None
        best = None
        best_score = -1
        for r in results:
            seg_dur = r.get("end", 0) - r.get("start", 0)
            if seg_dur >= duration * 0.8:
                score = 1.0 if abs(seg_dur - duration) < 2 else 0.5
                if score > best_score:
                    best_score = score
                    best = r
        return best or results[0]

    def get_outline(self, asset_id: str | None = None) -> dict:
        """Get a summary outline of indexed content."""
        indices = self.store.get_all_indices()
        if asset_id:
            indices = [i for i in indices if i.asset_id == asset_id]
        return {
            "n_assets": len(indices),
            "total_duration": sum(i.duration for i in indices),
            "assets": [i.summary() for i in indices],
        }
