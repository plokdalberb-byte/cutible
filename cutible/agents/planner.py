"""Planner agent — the Director (plan §7.1).

Analyzes the brief/prompt, creates a storyboard and edit plan,
determines structure, tone, pacing, and segment durations.

Uses LLM for intelligent planning when available, falls back to
deterministic heuristics when not.
"""

from __future__ import annotations

import json
from typing import Any

from .base import AgentMessage, BaseAgent, MessageType


class PlannerAgent(BaseAgent):
    """The Director/Planner — breaks down a brief into a structured edit plan.

    Input: brief/prompt + optional narrative index
    Output: structured edit plan with scenes, segments, timing, tone
    """

    def __init__(self, name: str = "planner", llm_client: Any | None = None):
        super().__init__(
            name=name,
            role="planner",
            description="Analyzes briefs and creates structured edit plans",
            llm_client=llm_client,
        )

    def execute(self, message: AgentMessage) -> AgentMessage:
        content = message.content
        brief = content.get("brief", "")
        narrative = content.get("narrative", {})
        target_duration = content.get("target_duration", 60.0)
        style = content.get("style", "informative")

        plan = self._create_plan(brief, narrative, target_duration, style)

        return self.send(
            to_agent=message.from_agent,
            msg_type=MessageType.RESULT,
            content={"plan": plan, "brief": brief},
            reply_to=message.id,
        )

    def _create_plan(self, brief: str, narrative: dict, target_duration: float, style: str) -> dict:
        """Create a structured edit plan from the brief."""
        # Try LLM first
        if self.llm and self.llm.available:
            llm_plan = self._plan_with_llm(brief, narrative, target_duration, style)
            if llm_plan is not None:
                return llm_plan

        # Fallback to heuristic
        return self._plan_heuristic(brief, narrative, target_duration, style)

    def _plan_with_llm(
        self, brief: str, narrative: dict, target_duration: float, style: str
    ) -> dict | None:
        """Use LLM to generate an intelligent edit plan."""
        assets_summary = json.dumps(narrative.get("assets", [])[:5], indent=2, default=str)[:3000]

        system_prompt = (
            "You are a professional video editor creating an edit plan. "
            "Analyze the brief and available source material, then return "
            "a JSON edit plan. Be specific about which assets to use and "
            "what time ranges to extract.\n\n"
            "Return ONLY a JSON object with this structure:\n"
            "{\n"
            '  "segments": [\n'
            '    {"type": "hook|body|outro", "description": "...", '
            '"source_asset": "asset_id", "src_start": 0.0, "src_end": 5.0, '
            '"timeline_start": 0.0, "timeline_end": 5.0, "rationale": "why this clip"}\n'
            "  ],\n"
            '  "tone": "energetic|calm|professional|humorous|informative",\n'
            '  "pacing": {"avg_clip_duration": 5.0, "transition_speed": "fast|medium|slow"},\n'
            '  "captions": [{"text": "...", "start": 0.0, "end": 3.0}],\n'
            '  "music_notes": "description of what music to use"\n'
            "}"
        )

        user_prompt = (
            f"Brief: {brief}\n"
            f"Target duration: {target_duration}s\n"
            f"Style: {style}\n"
            f"Available assets:\n{assets_summary}\n\n"
            "Create an edit plan that makes the best use of the available material."
        )

        result = self.llm.generate(system_prompt, user_prompt, temperature=0.7, max_tokens=2000)
        if result is None:
            return None

        # Validate and normalize LLM output
        segments = result.get("segments", [])
        if not segments:
            return None

        # Ensure required fields
        for seg in segments:
            seg.setdefault("type", "body")
            seg.setdefault("description", "")
            seg.setdefault("source_asset", None)
            seg.setdefault("src_start", 0.0)
            seg.setdefault("src_end", seg.get("src_start", 0) + 5.0)
            seg.setdefault("timeline_start", 0.0)
            seg.setdefault("timeline_end", seg.get("timeline_start", 0) + 5.0)
            seg.setdefault("rationale", "")

        # Convert to standard plan format
        normalized_segments = []
        for seg in segments:
            normalized_segments.append(
                {
                    "type": seg["type"],
                    "description": seg["description"],
                    "start": seg.get("timeline_start", 0.0),
                    "end": seg.get("timeline_end", seg.get("timeline_start", 0) + 5.0),
                    "source_asset": seg.get("source_asset"),
                    "src_start": seg.get("src_start", 0.0),
                    "src_end": seg.get("src_end", seg.get("src_start", 0) + 5.0),
                    "rationale": seg.get("rationale", ""),
                }
            )

        return {
            "brief": brief,
            "style": style,
            "target_duration": target_duration,
            "segments": normalized_segments,
            "tone": result.get("tone", style),
            "pacing": result.get("pacing", {}),
            "captions": result.get("captions", []),
            "music_notes": result.get("music_notes", ""),
            "structure": {
                "hook": normalized_segments[0] if normalized_segments else None,
                "body": [s for s in normalized_segments[1:-1]],
                "resolution": normalized_segments[-1] if len(normalized_segments) > 1 else None,
            },
            "source": "llm",
        }

    def _plan_heuristic(
        self, brief: str, narrative: dict, target_duration: float, style: str
    ) -> dict:
        """Deterministic heuristic plan generation (fallback)."""
        segments = self._plan_segments(brief, narrative, target_duration)

        return {
            "brief": brief,
            "style": style,
            "target_duration": target_duration,
            "segments": segments,
            "tone": self._determine_tone(brief, style),
            "pacing": self._determine_pacing(style, len(segments)),
            "structure": {
                "hook": segments[0] if segments else None,
                "body": segments[1:-1] if len(segments) > 2 else [],
                "resolution": segments[-1] if len(segments) > 1 else None,
            },
            "source": "heuristic",
        }

    def _plan_segments(self, brief: str, narrative: dict, target_duration: float) -> list[dict]:
        """Plan individual segments with timing."""
        assets = narrative.get("assets", [])

        if not assets:
            return [
                {
                    "type": "placeholder",
                    "description": brief,
                    "start": 0,
                    "end": target_duration,
                    "source_asset": None,
                }
            ]

        segments = []
        hook_duration = min(5.0, target_duration * 0.15)
        main_duration = target_duration - hook_duration
        n_assets = len(assets)
        per_asset = main_duration / max(n_assets, 1)

        segments.append(
            {
                "type": "hook",
                "description": "Opening hook to grab attention",
                "start": 0,
                "end": hook_duration,
                "source_asset": assets[0]["asset_id"] if assets else None,
            }
        )

        for i, asset in enumerate(assets):
            seg_start = hook_duration + i * per_asset
            seg_end = min(seg_start + per_asset, target_duration - 5.0)
            if seg_start >= target_duration - 5.0:
                break
            segments.append(
                {
                    "type": "body",
                    "description": f"Main content segment {i + 1}",
                    "start": seg_start,
                    "end": seg_end,
                    "source_asset": asset.get("asset_id"),
                    "duration": asset.get("duration", 0),
                }
            )

        segments.append(
            {
                "type": "outro",
                "description": "Closing / call to action",
                "start": target_duration - 5.0,
                "end": target_duration,
                "source_asset": assets[-1]["asset_id"] if assets else None,
            }
        )

        return segments

    def _determine_tone(self, brief: str, style: str) -> str:
        brief_lower = brief.lower()
        if any(w in brief_lower for w in ["energetic", "exciting", "fast"]):
            return "energetic"
        if any(w in brief_lower for w in ["calm", "relaxing", "peaceful"]):
            return "calm"
        if any(w in brief_lower for w in ["professional", "business", "corporate"]):
            return "professional"
        if any(w in brief_lower for w in ["funny", "humor", "comedy"]):
            return "humorous"
        return style

    def _determine_pacing(self, style: str, n_segments: int) -> dict:
        pacing_map = {
            "energetic": {"avg_clip_duration": 3.0, "transition_speed": "fast"},
            "calm": {"avg_clip_duration": 8.0, "transition_speed": "slow"},
            "professional": {"avg_clip_duration": 5.0, "transition_speed": "medium"},
            "humorous": {"avg_clip_duration": 4.0, "transition_speed": "fast"},
            "informative": {"avg_clip_duration": 6.0, "transition_speed": "medium"},
        }
        return pacing_map.get(style, pacing_map["informative"])
