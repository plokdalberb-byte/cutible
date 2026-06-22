"""Cutible Python SDK client.

Works in two modes:
1. In-process: directly uses Cutible's Python modules
2. HTTP client: connects to a Cutible REST API server

Usage::

    from cutible.sdk import CutibleClient

    # In-process mode
    client = CutibleClient()
    client.create_project("my_video", fps=30, width=1920, height=1080)
    client.add_asset("speaker", "video", uri="speaker.mp4", duration=60)
    client.add_track("v_main", "video")
    client.add_clip("v_main", "speaker", src_in=0, src_out=10)
    result = client.render("output.mp4")

    # HTTP mode
    client = CutibleClient(api_url="http://localhost:8000")
    client.create_project("my_video")
"""

from __future__ import annotations

import json

from ..compiler import FFmpegCompiler
from ..qc import run_qc
from ..schema import Project
from ..verbs import Editor


class CutibleClient:
    """High-level client for Cutible operations.

    Supports both in-process and HTTP-based operation.
    """

    def __init__(self, api_url: str | None = None):
        self.api_url = api_url.rstrip("/") if api_url else None
        self._editor: Editor | None = None
        self._project_id: str | None = None

        if not self.api_url:
            self._session: dict[str, Editor] = {}

    def create_project(
        self,
        project_id: str,
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        prompt: str = "",
    ) -> dict:
        if self.api_url:
            return self._http_post(
                "/projects",
                {
                    "id": project_id,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "prompt": prompt,
                },
            )
        project = Project(
            id=project_id,
            fps=fps,
            width=width,
            height=height,
            provenance={"prompt": prompt} if prompt else {},
        )
        self._editor = Editor(project)
        self._session[project_id] = self._editor
        self._project_id = project_id
        return {"created": project_id, "summary": project.summary()}

    def load_project(self, project_id: str, path: str) -> dict:
        if self.api_url:
            return self._http_post(f"/projects/{project_id}/load", {"path": path})
        project = Project.load(path)
        self._editor = Editor(project)
        self._session[project_id] = self._editor
        self._project_id = project_id
        return {"loaded": path, "summary": project.summary()}

    def read(self, zoom: str = "outline") -> dict:
        if self.api_url:
            return self._http_get(f"/projects/{self._project_id}", {"zoom": zoom})
        return self._editor.read(zoom)

    def add_asset(
        self,
        asset_id: str,
        asset_type: str,
        uri: str | None = None,
        duration: float | None = None,
        color: str = "black",
    ) -> dict:
        if self.api_url:
            return self._apply_verb(
                "add_asset",
                {
                    "asset_id": asset_id,
                    "type": asset_type,
                    "uri": uri,
                    "duration": duration,
                    "color": color,
                },
            )
        diff = self._editor.add_asset(asset_id, asset_type, uri=uri, duration=duration, color=color)
        return diff.to_dict()

    def add_track(self, track_id: str, kind: str) -> dict:
        if self.api_url:
            return self._apply_verb("add_track", {"track_id": track_id, "kind": kind})
        diff = self._editor.add_track(track_id, kind)
        return diff.to_dict()

    def add_clip(
        self,
        track_id: str,
        asset: str,
        src_in: float = 0,
        src_out: float | None = None,
        timeline_in: float | None = None,
        speed: float = 1.0,
        volume: float = 1.0,
        rationale: str | None = None,
    ) -> dict:
        if self.api_url:
            return self._apply_verb(
                "add_clip",
                {
                    "track_id": track_id,
                    "asset": asset,
                    "src_in": src_in,
                    "src_out": src_out,
                    "timeline_in": timeline_in,
                    "speed": speed,
                    "volume": volume,
                    "rationale": rationale,
                },
            )
        diff = self._editor.add_clip(
            track_id,
            asset,
            src_in=src_in,
            src_out=src_out,
            timeline_in=timeline_in,
            speed=speed,
            volume=volume,
            rationale=rationale,
        )
        return diff.to_dict()

    def trim(self, clip_id: str, src_in: float | None = None, src_out: float | None = None) -> dict:
        if self.api_url:
            return self._apply_verb(
                "trim",
                {
                    "clip_id": clip_id,
                    "src_in": src_in,
                    "src_out": src_out,
                },
            )
        diff = self._editor.trim(clip_id, src_in=src_in, src_out=src_out)
        return diff.to_dict()

    def move(self, clip_id: str, timeline_in: float) -> dict:
        if self.api_url:
            return self._apply_verb(
                "move",
                {
                    "clip_id": clip_id,
                    "timeline_in": timeline_in,
                },
            )
        diff = self._editor.move(clip_id, timeline_in)
        return diff.to_dict()

    def split(self, clip_id: str, t: float) -> dict:
        if self.api_url:
            return self._apply_verb("split", {"clip_id": clip_id, "t": t})
        diff = self._editor.split(clip_id, t)
        return diff.to_dict()

    def ripple_delete(self, clip_id: str) -> dict:
        if self.api_url:
            return self._apply_verb("ripple_delete", {"clip_id": clip_id})
        diff = self._editor.ripple_delete(clip_id)
        return diff.to_dict()

    def add_transition(self, clip_id: str, kind: str = "in", duration: float = 0.5) -> dict:
        if self.api_url:
            return self._apply_verb(
                "add_transition",
                {
                    "clip_id": clip_id,
                    "kind": kind,
                    "duration": duration,
                },
            )
        diff = self._editor.add_transition(clip_id, kind, duration)
        return diff.to_dict()

    def add_text_layer(
        self, track_id: str, text: str, timeline_in: float, timeline_out: float, **style
    ) -> dict:
        if self.api_url:
            return self._apply_verb(
                "add_text_layer",
                {
                    "track_id": track_id,
                    "text": text,
                    "timeline_in": timeline_in,
                    "timeline_out": timeline_out,
                    **style,
                },
            )
        diff = self._editor.add_text_layer(track_id, text, timeline_in, timeline_out, **style)
        return diff.to_dict()

    def add_audio(
        self,
        asset: str,
        src_in: float = 0,
        src_out: float | None = None,
        timeline_in: float = 0,
        volume: float = 1.0,
        track_id: str = "music",
    ) -> dict:
        if self.api_url:
            return self._apply_verb(
                "add_audio",
                {
                    "asset": asset,
                    "src_in": src_in,
                    "src_out": src_out,
                    "timeline_in": timeline_in,
                    "volume": volume,
                    "track_id": track_id,
                },
            )
        diff = self._editor.add_audio(
            asset,
            src_in=src_in,
            src_out=src_out,
            timeline_in=timeline_in,
            volume=volume,
            track_id=track_id,
        )
        return diff.to_dict()

    def set_speed(self, clip_id: str, speed: float) -> dict:
        if self.api_url:
            return self._apply_verb("set_speed", {"clip_id": clip_id, "speed": speed})
        diff = self._editor.set_speed(clip_id, speed)
        return diff.to_dict()

    def set_volume(self, clip_id: str, volume: float) -> dict:
        if self.api_url:
            return self._apply_verb("set_volume", {"clip_id": clip_id, "volume": volume})
        diff = self._editor.set_volume(clip_id, volume)
        return diff.to_dict()

    def checkpoint(self, label: str = "") -> dict:
        if self.api_url:
            return self._apply_verb("checkpoint", {"label": label})
        diff = self._editor.checkpoint(label)
        return diff.to_dict()

    def undo(self) -> dict:
        if self.api_url:
            return self._apply_verb("undo", {})
        diff = self._editor.undo()
        return diff.to_dict()

    def render(self, output: str, run_qc: bool = True) -> dict:
        if self.api_url:
            return self._http_post(
                f"/projects/{self._project_id}/render",
                {
                    "output": output,
                    "run_qc": run_qc,
                },
            )
        compiler = FFmpegCompiler(self._editor.project)
        result = compiler.render(output)
        if run_qc:
            report = run_qc(
                output,
                expected_duration=self._editor.project.duration,
                loudness_target=self._editor.project.globals.loudness_target,
            )
            result["qc"] = report.to_dict()
        return result

    def qc(self, file: str, expected_duration: float | None = None) -> dict:
        if self.api_url:
            return self._http_post(
                "/qc",
                {
                    "file": file,
                    "expected_duration": expected_duration,
                },
            )
        report = run_qc(file, expected_duration=expected_duration)
        return report.to_dict()

    def save(self, path: str) -> dict:
        if self.api_url:
            return self._http_post(f"/projects/{self._project_id}/save", {"path": path})
        self._editor.project.save(path)
        return {"saved": path, "hash": self._editor.project.content_hash()}

    def export_otio(self, output_path: str) -> dict:
        from ..otio_bridge import OTIOExporter

        if self.api_url:
            return self._http_get(
                f"/projects/{self._project_id}/otio", {"output_path": output_path}
            )
        exporter = OTIOExporter(self._editor.project)
        return exporter.export(output_path)

    def import_otio(self, otio_path: str, project_id: str | None = None) -> dict:
        from ..otio_bridge import OTIOImporter

        if self.api_url:
            return self._http_post(
                f"/projects/{self._project_id}/otio/import", {"otio_path": otio_path}
            )
        importer = OTIOImporter()
        project = importer.import_file(otio_path, project_id)
        self._editor = Editor(project)
        self._project_id = project.id
        return {"imported": otio_path, "summary": project.summary()}

    def run_agent(
        self,
        brief: str,
        target_duration: float = 60.0,
        style: str = "informative",
        max_iterations: int = 3,
        index_dir: str = ".cutible/index",
    ) -> dict:
        if self.api_url:
            return self._http_post(
                "/agent/run",
                {
                    "brief": brief,
                    "target_duration": target_duration,
                    "style": style,
                    "max_iterations": max_iterations,
                    "index_dir": index_dir,
                },
            )
        import os

        from ..agents.orchestrator import Orchestrator

        openai_key = os.environ.get("OPENAI_API_KEY")
        openai_base = os.environ.get("OPENAI_BASE_URL")
        openai_model = os.environ.get("OPENAI_MODEL")
        orchestrator = Orchestrator(
            max_iterations=max_iterations,
            openai_api_key=openai_key,
            openai_base_url=openai_base,
            openai_model=openai_model,
        )
        return orchestrator.run(
            brief=brief,
            target_duration=target_duration,
            style=style,
            index_dir=index_dir,
        )

    def _apply_verb(self, verb: str, args: dict) -> dict:
        return self._http_post(
            f"/projects/{self._project_id}/verbs",
            {
                "verb": verb,
                "args": args,
            },
        )

    def _http_get(self, path: str, params: dict | None = None) -> dict:
        import urllib.request

        url = f"{self.api_url}{path}"
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
            url += f"?{query}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def _http_post(self, path: str, data: dict) -> dict:
        import urllib.request

        url = f"{self.api_url}{path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
