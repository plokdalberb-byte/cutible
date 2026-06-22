"""Orchestrator — coordinates the agent swarm (plan §7.1).

Manages the workflow pipeline: Plan → Edit → Sound → QC → Iterate.
Handles message routing, state management, and convergence detection.
"""

from __future__ import annotations

import json
import logging

from .base import AgentMessage, BaseAgent, MessageType
from .editor import EditorAgent
from .planner import PlannerAgent
from .qc_agent import QCAgent
from .sound import SoundAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the multi-agent swarm for video editing.

    Manages the end-to-end pipeline:
    1. Planner creates the edit plan
    2. Editor assembles the timeline
    3. Sound engineer handles audio
    4. QC reviewer checks quality
    5. If issues found, loop back to editor with feedback
    """

    def __init__(
        self,
        max_iterations: int = 3,
        vlm_model: str = "gemini",
        vlm_api_key: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str | None = None,
    ):
        self.max_iterations = max_iterations
        self.agents: dict[str, BaseAgent] = {}
        self.message_log: list[dict] = []
        self.current_iteration = 0

        from .llm_client import LLMClient

        self.llm_client = LLMClient(
            provider="openai",
            api_key=openai_api_key,
            base_url=openai_base_url,
            model=openai_model,
        )

        self._register_agent(PlannerAgent("planner", llm_client=self.llm_client))
        self._register_agent(EditorAgent("editor", llm_client=self.llm_client))
        self._register_agent(SoundAgent("sound", llm_client=self.llm_client))
        self._register_agent(QCAgent("qc_reviewer", vlm_model, vlm_api_key))

    def _register_agent(self, agent: BaseAgent) -> None:
        self.agents[agent.name] = agent

    def run(
        self,
        brief: str,
        narrative: dict | None = None,
        target_duration: float = 60.0,
        style: str = "informative",
        music_assets: list | None = None,
        skip_vlm: bool = False,
        index_dir: str | None = None,
    ) -> dict:
        """Run the full editing pipeline.

        If ``narrative`` is None and ``index_dir`` is provided, loads the
        NarrativeIndex from disk and converts it to the agent-compatible format.
        Returns the final project state and all intermediate results.
        """
        if narrative is None:
            narrative = self._load_narrative(index_dir or ".cutible/index")

        results = {
            "brief": brief,
            "iterations": [],
            "final_project": None,
            "passed": False,
        }

        project_data = None
        edit_plan = None

        for iteration in range(self.max_iterations):
            self.current_iteration = iteration
            logger.info(f"=== Iteration {iteration + 1}/{self.max_iterations} ===")

            iter_result = {"iteration": iteration + 1}

            if iteration == 0 or edit_plan is None:
                edit_plan = self._run_planner(brief, narrative, target_duration, style)
                iter_result["plan"] = edit_plan

            project_data = self._run_editor(edit_plan, narrative, project_data)
            iter_result["project_duration"] = self._get_duration(project_data)

            project_data = self._run_sound(project_data, music_assets or [], "v_main", -14.0)

            qc_result = self._run_qc(project_data, None, brief, skip_vlm)
            iter_result["qc"] = qc_result

            if qc_result.get("passed", False):
                results["passed"] = True
                results["final_project"] = project_data
                iter_result["status"] = "passed"
                results["iterations"].append(iter_result)
                break

            if iteration < self.max_iterations - 1:
                suggestions = qc_result.get("suggestions", [])
                if suggestions:
                    edit_plan = self._refine_plan(edit_plan, suggestions, qc_result)
                    iter_result["refined_plan"] = True
                iter_result["status"] = "needs_revision"
            else:
                results["final_project"] = project_data
                iter_result["status"] = "max_iterations_reached"

            results["iterations"].append(iter_result)

        return results

    def _run_planner(self, brief: str, narrative: dict, target_duration: float, style: str) -> dict:
        planner = self.agents["planner"]
        msg = AgentMessage(
            from_agent="orchestrator",
            to_agent="planner",
            type=MessageType.TASK,
            content={
                "brief": brief,
                "narrative": narrative,
                "target_duration": target_duration,
                "style": style,
            },
        )
        planner.receive(msg)
        responses = planner.process_all()
        if responses:
            return responses[0].content.get("plan", {})
        return {}

    def _run_editor(self, plan: dict, narrative: dict, project_data: dict | None) -> dict:
        editor = self.agents["editor"]
        msg = AgentMessage(
            from_agent="orchestrator",
            to_agent="editor",
            type=MessageType.TASK,
            content={
                "plan": plan,
                "narrative": narrative,
                "project": project_data,
            },
        )
        editor.receive(msg)
        responses = editor.process_all()
        if responses:
            return responses[0].content.get("project", {})
        return project_data or {}

    def _run_sound(
        self, project_data: dict, music_assets: list, voice_track: str, target_lufs: float
    ) -> dict:
        sound = self.agents["sound"]
        msg = AgentMessage(
            from_agent="orchestrator",
            to_agent="sound",
            type=MessageType.TASK,
            content={
                "project": project_data,
                "music_assets": music_assets,
                "voice_track": voice_track,
                "target_lufs": target_lufs,
            },
        )
        sound.receive(msg)
        responses = sound.process_all()
        if responses:
            return responses[0].content.get("project", {})
        return project_data

    def _run_qc(
        self, project_data: dict, render_path: str | None, brief: str, skip_vlm: bool
    ) -> dict:
        qc = self.agents["qc_reviewer"]
        msg = AgentMessage(
            from_agent="orchestrator",
            to_agent="qc_reviewer",
            type=MessageType.TASK,
            content={
                "project": project_data,
                "render_path": render_path,
                "brief": brief,
                "skip_vlm": skip_vlm,
            },
        )
        qc.receive(msg)
        responses = qc.process_all()
        if responses:
            return responses[0].content
        return {"passed": False}

    def _refine_plan(self, plan: dict, suggestions: list[str], qc_result: dict) -> dict:
        """Refine the edit plan based on QC feedback."""
        refined = json.loads(json.dumps(plan))
        for seg in refined.get("segments", []):
            seg["refinement_notes"] = suggestions[:3]
        return refined

    def _get_duration(self, project_data: dict | None) -> float:
        if not project_data:
            return 0.0
        tracks = project_data.get("tracks", [])
        max_end = 0.0
        for track in tracks:
            for clip in track.get("clips", []):
                end = clip.get("timeline_in", 0) + (
                    clip.get("src_out", 0) - clip.get("src_in", 0)
                ) / clip.get("speed", 1.0)
                max_end = max(max_end, end)
        return max_end

    def _load_narrative(self, index_dir: str) -> dict:
        """Load NarrativeIndex from disk and convert to agent format."""
        try:
            from ..index.store import IndexStore

            store = IndexStore(index_dir)
            narrative = store.load_narrative()
            if narrative is not None:
                return narrative.to_agent_dict()
            # Fallback: build from individual asset indices
            narrative = store.build_narrative("agent_session")
            return narrative.to_agent_dict()
        except Exception as e:
            logger.warning(f"Could not load narrative from {index_dir}: {e}")
            return {"assets": [], "total_duration": 0.0}

    def get_state(self) -> dict:
        return {
            "iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "agents": {name: agent.get_state() for name, agent in self.agents.items()},
            "message_log_size": len(self.message_log),
        }
