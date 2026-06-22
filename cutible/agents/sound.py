"""Sound agent — the AUDIO ENGINEER (plan §7.1).

Handles music selection, ducking, beat-sync, LUFS normalization,
and overall sound design.

Uses LLM for intelligent audio decisions when available.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .base import BaseAgent, AgentMessage, MessageType


class SoundAgent(BaseAgent):
    """The Sound Engineer — handles all audio aspects of the edit.

    Input: project state + sound requirements
    Output: project with audio mixed, ducked, normalized
    """

    def __init__(self, name: str = "sound", llm_client: Optional[Any] = None):
        super().__init__(
            name=name,
            role="sound_engineer",
            description="Handles music, ducking, beat-sync, and LUFS normalization",
            llm_client=llm_client,
        )

    def execute(self, message: AgentMessage) -> AgentMessage:
        content = message.content
        project_data = content.get("project")
        music_assets = content.get("music_assets", [])
        voice_track = content.get("voice_track", "v_main")
        target_lufs = content.get("target_lufs", -14.0)

        from ..schema import Project
        from ..verbs import Editor as EditorClass
        from ..verbs_high import HighLevelVerbs

        project = Project.model_validate(project_data) if project_data else Project(id="sound_edit")
        editor = EditorClass(project)
        high_level = HighLevelVerbs(editor)

        actions = self._apply_sound(editor, high_level, music_assets,
                                     voice_track, target_lufs)

        return self.send(
            to_agent=message.from_agent,
            msg_type=MessageType.RESULT,
            content={
                "project": editor.project.model_dump(mode="json"),
                "actions": actions,
            },
            reply_to=message.id,
        )

    def _apply_sound(self, editor, high_level: HighLevelVerbs,
                     music_assets: list, voice_track: str,
                     target_lufs: float) -> list[dict]:
        """Apply sound processing to the project."""
        actions = []
        has_voice = any(
            t.id == voice_track and t.clips
            for t in editor.project.tracks
        )

        # Use LLM to determine optimal sound settings if available
        sound_params = self._plan_sound_with_llm(editor, music_assets, has_voice)

        for music in music_assets:
            asset_id = music.get("asset_id")
            volume = music.get("volume", sound_params.get("music_volume", 0.2))
            track_id = music.get("track_id", "music")

            if not any(a.id == asset_id for a in editor.project.assets):
                continue

            if not any(t.id == track_id for t in editor.project.tracks):
                editor.add_track(track_id, "audio")

            diff = editor.add_audio(
                asset_id,
                volume=volume,
                track_id=track_id,
                rationale="music bed",
            )
            actions.append({"verb": "add_audio", "diff": diff.to_dict()})

        if has_voice and any(
            t.id == "music" and t.clips
            for t in editor.project.tracks
        ):
            duck_level = sound_params.get("duck_level", 0.15)
            try:
                diff = high_level.auto_ducking(voice_track, "music",
                                               duck_level=duck_level)
                actions.append({"verb": "auto_ducking", "diff": diff.to_dict()})
            except Exception:
                pass

        editor.project.globals.loudness_target = sound_params.get("lufs", target_lufs)
        actions.append({
            "verb": "set_loudness_target",
            "target_lufs": sound_params.get("lufs", target_lufs),
        })

        return actions

    def _plan_sound_with_llm(self, editor, music_assets: list,
                              has_voice: bool) -> dict:
        """Use LLM to determine optimal sound parameters."""
        if not self.llm or not self.llm.available:
            return {
                "music_volume": 0.2,
                "duck_level": 0.15,
                "lufs": -14.0,
            }

        # Gather project context
        tracks_summary = []
        for track in editor.project.tracks:
            tracks_summary.append({
                "id": track.id,
                "kind": track.kind.value,
                "n_clips": len(track.clips),
                "duration": track.duration,
            })

        system_prompt = (
            "You are a sound engineer for video editing. Given the project "
            "state, determine optimal audio parameters.\n\n"
            "Return ONLY a JSON object:\n"
            "{\n"
            '  "music_volume": 0.0-1.0,\n'
            '  "duck_level": 0.0-1.0,\n'
            '  "lufs": -24.0 to -10.0,\n'
            '  "notes": "brief explanation"\n'
            "}"
        )

        user_prompt = (
            f"Project tracks: {json.dumps(tracks_summary, indent=2)}\n"
            f"Has voice track: {has_voice}\n"
            f"Number of music assets: {len(music_assets)}\n\n"
            "Determine optimal sound parameters."
        )

        result = self.llm.generate(system_prompt, user_prompt,
                                    temperature=0.3, max_tokens=500)
        if result is None:
            return {
                "music_volume": 0.2,
                "duck_level": 0.15,
                "lufs": -14.0,
            }

        return {
            "music_volume": result.get("music_volume", 0.2),
            "duck_level": result.get("duck_level", 0.15),
            "lufs": result.get("lufs", -14.0),
        }
