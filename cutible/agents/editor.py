"""Editor agent — the MONTAGEUR (plan §7.1).

Takes the edit plan from the planner, selects clips from the index,
applies verb operations, and builds the actual timeline.

Uses LLM for intelligent clip selection and timing when available.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .base import BaseAgent, AgentMessage, MessageType


class EditorAgent(BaseAgent):
    """The Editor/Montageur — executes the edit plan using verb primitives.

    Input: edit plan + narrative index + editor session
    Output: assembled timeline (project state)
    """

    def __init__(self, name: str = "editor", llm_client: Optional[Any] = None):
        super().__init__(
            name=name,
            role="editor",
            description="Selects clips and builds the timeline from an edit plan",
            llm_client=llm_client,
        )

    def execute(self, message: AgentMessage) -> AgentMessage:
        content = message.content
        plan = content.get("plan", {})
        narrative = content.get("narrative", {})
        project_data = content.get("project")

        from ..schema import Project
        from ..verbs import Editor as EditorClass
        from ..verbs_high import HighLevelVerbs

        if project_data:
            project = Project.model_validate(project_data)
        else:
            project = Project(id="agent_edit")

        editor = EditorClass(project)
        high = HighLevelVerbs(editor)

        actions = self._execute_plan(editor, high, plan, narrative)

        return self.send(
            to_agent=message.from_agent,
            msg_type=MessageType.RESULT,
            content={
                "project": editor.project.model_dump(mode="json"),
                "actions": actions,
                "duration": editor.project.duration,
            },
            reply_to=message.id,
        )

    def _execute_plan(self, editor, high_level: HighLevelVerbs,
                      plan: dict, narrative: dict) -> list[dict]:
        """Execute the edit plan step by step."""
        actions = []
        segments = plan.get("segments", [])

        narrative_assets = narrative.get("assets", [])
        asset_map = {a["asset_id"]: a for a in narrative_assets}

        # Register assets from narrative
        for asset_info in narrative_assets:
            asset_id = asset_info.get("asset_id")
            if asset_id and not any(a.id == asset_id for a in editor.project.assets):
                uri = asset_info.get("uri", "")
                duration = asset_info.get("duration")
                if uri:
                    editor.add_asset(asset_id, "video", uri=uri, duration=duration)
                else:
                    editor.add_asset(asset_id, "color", color="gray",
                                    duration=duration if duration else 10.0)

        # Use LLM to refine clip selection if available
        if self.llm and self.llm.available and narrative_assets:
            refined_segments = self._refine_with_llm(plan, narrative_assets)
            if refined_segments:
                segments = refined_segments

        editor.add_track("v_main", "video")
        editor.add_track("captions", "caption")

        for segment in segments:
            seg_type = segment.get("type", "body")
            source_asset = segment.get("source_asset")
            start = segment.get("start", 0)
            end = segment.get("end", start + 5)
            duration = end - start

            if source_asset and source_asset in asset_map:
                asset_info = asset_map[source_asset]
                src_start = segment.get("src_start", asset_info.get("start", 0))
                src_end = segment.get("src_end", src_start + duration)

                diff = editor.add_clip(
                    "v_main", source_asset,
                    src_in=src_start,
                    src_out=src_end,
                    timeline_in=start,
                    rationale=segment.get("rationale",
                        f"{seg_type}: {segment.get('description', '')}"),
                )
                actions.append({
                    "verb": "add_clip",
                    "segment": segment.get("description"),
                    "diff": diff.to_dict(),
                })

            # Add captions if provided
            captions = plan.get("captions", [])
            for cap in captions:
                if cap.get("start", 0) >= start and cap.get("end", end) <= end:
                    diff = editor.add_text_layer(
                        "captions", cap["text"],
                        cap.get("start", start), cap.get("end", end),
                    )
                    actions.append({
                        "verb": "add_text_layer",
                        "diff": diff.to_dict(),
                    })

        # Add transitions between clips
        track = editor.project.track("v_main")
        for clip in track.clips:
            diff = editor.add_transition(clip.id, "in", 0.3)
            actions.append({"verb": "add_transition", "diff": diff.to_dict()})

        return actions

    def _refine_with_llm(self, plan: dict,
                         narrative_assets: list) -> Optional[list[dict]]:
        """Use LLM to refine clip selection and timing."""
        assets_summary = json.dumps(narrative_assets[:5], indent=2,
                                     default=str)[:3000]
        plan_summary = json.dumps(plan, indent=2, default=str)[:2000]

        system_prompt = (
            "You are a professional video editor. Given an edit plan and "
            "available source assets, refine the clip selections with "
            "specific time ranges. For each segment, decide which asset "
            "to use and what src_in/src_out range to extract.\n\n"
            "Return ONLY a JSON array of segments:\n"
            "[\n"
            '  {"type": "hook|body|outro", "description": "...", '
            '"source_asset": "asset_id", "src_start": 0.0, "src_end": 5.0, '
            '"start": 0.0, "end": 5.0, "rationale": "why this clip at this time"}\n'
            "]"
        )

        user_prompt = (
            f"Edit plan:\n{plan_summary}\n\n"
            f"Available assets:\n{assets_summary}\n\n"
            "Refine the clip selections with specific time ranges."
        )

        result = self.llm.generate(system_prompt, user_prompt,
                                    temperature=0.5, max_tokens=2000)
        if result is None:
            return None

        # Handle both list and dict responses
        if isinstance(result, dict):
            segments = result.get("segments", [])
        elif isinstance(result, list):
            segments = result
        else:
            return None

        if not segments:
            return None

        # Normalize segments
        for seg in segments:
            seg.setdefault("type", "body")
            seg.setdefault("description", "")
            seg.setdefault("source_asset", None)
            seg.setdefault("src_start", seg.get("start", 0))
            seg.setdefault("src_end", seg.get("end", seg.get("start", 0) + 5))
            seg.setdefault("start", 0.0)
            seg.setdefault("end", seg.get("start", 0) + 5.0)
            seg.setdefault("rationale", "")

        return segments
