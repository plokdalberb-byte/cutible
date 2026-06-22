"""Cutible MCP server — the PRIMARY agent interface (plan §8.1).

Any MCP-speaking agent (Claude, GPT, custom) connects Cutible as a set of
tools and edits video the agent-native way: call a verb, read the diff, read
the timeline, render a proxy, run QC, iterate.

Transport: newline-delimited JSON-RPC 2.0 over stdio (the MCP stdio transport),
implemented with zero third-party dependencies so it runs anywhere Python does.

Run:
    python -m cutible.mcp_server
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Optional

from .schema import Project
from .verbs import Editor, VerbError
from .compiler import FFmpegCompiler
from .qc import run_qc

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "cutible", "version": "0.1.0"}

# A single editing session per server process (embeddable pipeline model).
_SESSION: dict[str, Optional[Editor]] = {"editor": None}


def _ed() -> Editor:
    if _SESSION["editor"] is None:
        raise VerbError("no active project",
                        hint="call create_project (or load_project) first")
    return _SESSION["editor"]


# --------------------------------------------------------------------------- #
# tool implementations -> all return JSON-able dicts
# --------------------------------------------------------------------------- #
def t_create_project(id: str, fps: int = 30, width: int = 1920, height: int = 1080,
                     prompt: str = "") -> dict:
    p = Project(id=id, fps=fps, width=width, height=height,
                provenance={"prompt": prompt} if prompt else {})
    _SESSION["editor"] = Editor(p)
    return {"created": id, "summary": p.summary()}


def t_load_project(path: str) -> dict:
    _SESSION["editor"] = Editor(Project.load(path))
    return {"loaded": path, "summary": _ed().project.summary()}


def t_save_project(path: str) -> dict:
    _ed().project.save(path)
    return {"saved": path, "content_hash": _ed().project.content_hash()}


def t_read(zoom: str = "outline") -> dict:
    return _ed().read(zoom)


def t_render(output: str, run_qc_after: bool = True, dry_run: bool = False) -> dict:
    comp = FFmpegCompiler(_ed().project)
    res = comp.render(output, dry_run=dry_run)
    if run_qc_after and not dry_run:
        report = run_qc(output, expected_duration=_ed().project.duration,
                        loudness_target=_ed().project.globals.loudness_target)
        res["qc"] = report.to_dict()
    return res


def t_qc(file: str, expected_duration: Optional[float] = None) -> dict:
    return run_qc(file, expected_duration=expected_duration).to_dict()


# --- ingest pipeline ------------------------------------------------------- #
def t_ingest_asset(asset_id: str, uri: str, index_dir: str = ".cutible/index") -> dict:
    from .ingest import IngestPipeline
    from .ingest.pipeline import IngestConfig
    pipeline = IngestPipeline(IngestConfig(index_dir=index_dir))
    result = pipeline.ingest_asset(asset_id, uri)
    return result.to_dict()


def t_build_narrative(project_id: str = "default",
                      index_dir: str = ".cutible/index") -> dict:
    from .ingest import IngestPipeline
    from .ingest.pipeline import IngestConfig
    pipeline = IngestPipeline(IngestConfig(index_dir=index_dir))
    narrative = pipeline.build_narrative(project_id)
    return narrative.summary()


def t_search_index(query: str, index_dir: str = ".cutible/index") -> dict:
    from .index import IndexStore, IndexSearcher
    store = IndexStore(index_dir)
    searcher = IndexSearcher(store)
    return {"results": searcher.search_text(query)[:20]}


# --- high-level verbs ------------------------------------------------------ #
def t_remove_silences(track_id: str, min_silence: float = 0.5) -> dict:
    from .verbs_high import HighLevelVerbs
    hl = HighLevelVerbs(_ed())
    diff = hl.remove_silences(track_id, min_silence=min_silence)
    return diff.to_dict()


def t_reframe_to(asset_id: str, target_aspect: str = "9:16",
                 focus: str = "center") -> dict:
    from .verbs_high import HighLevelVerbs
    hl = HighLevelVerbs(_ed())
    diff = hl.reframe_to(asset_id, target_aspect, focus)
    return diff.to_dict()


def t_sync_cuts_to_beat(track_id: str, audio_track_id: str) -> dict:
    from .verbs_high import HighLevelVerbs
    hl = HighLevelVerbs(_ed())
    diff = hl.sync_cuts_to_beat(track_id, audio_track_id)
    return diff.to_dict()


def t_generate_captions(track_id: str, style: str = "default",
                        index_dir: str = ".cutible/index") -> dict:
    from .verbs_high import HighLevelVerbs
    from .index import IndexStore
    store = IndexStore(index_dir)
    hl = HighLevelVerbs(_ed(), index_store=store)
    diff = hl.generate_captions(track_id, style)
    return diff.to_dict()


def t_auto_ducking(voice_track_id: str, music_track_id: str,
                   duck_level: float = 0.15) -> dict:
    from .verbs_high import HighLevelVerbs
    hl = HighLevelVerbs(_ed())
    diff = hl.auto_ducking(voice_track_id, music_track_id, duck_level)
    return diff.to_dict()


def t_assemble_from_plan(plan_json: str) -> dict:
    from .verbs_high import HighLevelVerbs
    hl = HighLevelVerbs(_ed())
    plan = json.loads(plan_json) if isinstance(plan_json, str) else plan_json
    diff = hl.assemble_from_plan(plan)
    return diff.to_dict()


def t_make_short(topic: str, duration: float = 60.0,
                 index_dir: str = ".cutible/index") -> dict:
    from .verbs_high import HighLevelVerbs
    from .index import IndexStore
    store = IndexStore(index_dir)
    hl = HighLevelVerbs(_ed(), index_store=store)
    diff = hl.make_short(topic, duration)
    return diff.to_dict()


# --- perception / VLM review ---------------------------------------------- #
def t_vlm_review(video_path: str, brief: str = "",
                 model: str = "gemini") -> dict:
    from .perception import VLMReview
    reviewer = VLMReview(model=model)
    report = reviewer.review(video_path, brief=brief)
    return report.to_dict()


def t_render_proxy(output: str, width: int = 480, height: int = 270) -> dict:
    from .perception import ProxyRenderer
    from .perception.proxy_render import ProxyConfig
    renderer = ProxyRenderer(ProxyConfig(width=width, height=height))
    return renderer.render_proxy(_ed().project, output)


# --- agents ---------------------------------------------------------------- #
def t_run_agent_swarm(brief: str, target_duration: float = 60.0,
                      style: str = "informative",
                      max_iterations: int = 3,
                      index_dir: str = ".cutible/index") -> dict:
    import os
    from .agents.orchestrator import Orchestrator
    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base = os.environ.get("OPENAI_BASE_URL")
    openai_model = os.environ.get("OPENAI_MODEL")
    orchestrator = Orchestrator(
        max_iterations=max_iterations,
        openai_api_key=openai_key,
        openai_base_url=openai_base,
        openai_model=openai_model,
    )
    return orchestrator.run(brief=brief, target_duration=target_duration,
                            style=style, index_dir=index_dir)


# --- OTIO bridge ----------------------------------------------------------- #
def t_export_otio(output_path: str) -> dict:
    from .otio_bridge import OTIOExporter
    exporter = OTIOExporter(_ed().project)
    return exporter.export(output_path)


def t_import_otio(otio_path: str) -> dict:
    from .otio_bridge import OTIOImporter
    importer = OTIOImporter()
    project = importer.import_file(otio_path)
    _SESSION["editor"] = Editor(project)
    return {"imported": otio_path, "summary": project.summary()}


# --- render farm ----------------------------------------------------------- #
def t_render_farm(output: str, n_workers: int = 2) -> dict:
    from .render_farm import RenderFarmManager
    farm = RenderFarmManager(n_workers=n_workers)
    return farm.render(_ed().project, output)


def t_render_farm_dry_run() -> dict:
    from .render_farm import RenderFarmManager
    farm = RenderFarmManager()
    return farm.render_dry_run(_ed().project)


# verbs that proxy straight to the Editor and return their diff
_VERB_METHODS = [
    "add_asset", "add_track", "add_clip", "trim", "move", "split",
    "ripple_delete", "set_speed", "set_volume", "add_transition",
    "add_text_layer", "add_audio", "checkpoint", "undo",
]


def _make_verb(name: str):
    def _call(**kwargs) -> dict:
        diff = getattr(_ed(), name)(**kwargs)
        return diff.to_dict()
    return _call


TOOLS: dict[str, Any] = {
    "create_project": t_create_project,
    "load_project": t_load_project,
    "save_project": t_save_project,
    "read": t_read,
    "render": t_render,
    "qc": t_qc,
    # --- ingest pipeline ---
    "ingest_asset": t_ingest_asset,
    "build_narrative": t_build_narrative,
    "search_index": t_search_index,
    # --- high-level verbs ---
    "remove_silences": t_remove_silences,
    "reframe_to": t_reframe_to,
    "sync_cuts_to_beat": t_sync_cuts_to_beat,
    "generate_captions": t_generate_captions,
    "auto_ducking": t_auto_ducking,
    "assemble_from_plan": t_assemble_from_plan,
    "make_short": t_make_short,
    # --- perception / VLM review ---
    "vlm_review": t_vlm_review,
    "render_proxy": t_render_proxy,
    # --- agents ---
    "run_agent_swarm": t_run_agent_swarm,
    # --- OTIO bridge ---
    "export_otio": t_export_otio,
    "import_otio": t_import_otio,
    # --- render farm ---
    "render_farm": t_render_farm,
    "render_farm_dry_run": t_render_farm_dry_run,
}
for _m in _VERB_METHODS:
    TOOLS[_m] = _make_verb(_m)


# --------------------------------------------------------------------------- #
# tool schemas (advertised to the agent)
# --------------------------------------------------------------------------- #
def _obj(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required}


_S = lambda **k: {"type": "string", **k}        # noqa: E731
_N = lambda **k: {"type": "number", **k}        # noqa: E731
_I = lambda **k: {"type": "integer", **k}       # noqa: E731
_B = lambda **k: {"type": "boolean", **k}       # noqa: E731

TOOL_SCHEMAS = {
    "create_project": (_obj({"id": _S(), "fps": _I(), "width": _I(), "height": _I(),
                             "prompt": _S()}, ["id"]),
                       "Start a new Timeline-as-Data project."),
    "load_project": (_obj({"path": _S()}, ["path"]), "Load a project from JSON."),
    "save_project": (_obj({"path": _S()}, ["path"]), "Save the project to JSON."),
    "read": (_obj({"zoom": _S(enum=["summary", "outline", "detail"])}, []),
             "Read the timeline at a zoom level (the agent inspects its work)."),
    "add_asset": (_obj({"asset_id": _S(), "type": _S(enum=["video", "audio", "image", "color"]),
                        "uri": _S(), "duration": _N(), "color": _S()}, ["asset_id", "type"]),
                  "Register a source asset."),
    "add_track": (_obj({"track_id": _S(), "kind": _S(enum=["video", "audio", "caption"])},
                       ["track_id", "kind"]), "Add a track."),
    "add_clip": (_obj({"track_id": _S(), "asset": _S(), "src_in": _N(), "src_out": _N(),
                       "timeline_in": _N(), "speed": _N(), "volume": _N(),
                       "rationale": _S()}, ["track_id", "asset"]),
                 "Place a slice of an asset on a track (returns a diff)."),
    "trim": (_obj({"clip_id": _S(), "src_in": _N(), "src_out": _N()}, ["clip_id"]),
             "Adjust a clip's source in/out."),
    "move": (_obj({"clip_id": _S(), "timeline_in": _N()}, ["clip_id", "timeline_in"]),
             "Move a clip on its track."),
    "split": (_obj({"clip_id": _S(), "t": _N()}, ["clip_id", "t"]),
              "Split a clip at timeline time t."),
    "ripple_delete": (_obj({"clip_id": _S()}, ["clip_id"]),
                      "Delete a clip and close the gap."),
    "set_speed": (_obj({"clip_id": _S(), "speed": _N()}, ["clip_id", "speed"]),
                  "Set clip playback speed."),
    "set_volume": (_obj({"clip_id": _S(), "volume": _N()}, ["clip_id", "volume"]),
                   "Set clip volume."),
    "add_transition": (_obj({"clip_id": _S(), "kind": _S(enum=["in", "out"]), "duration": _N()},
                            ["clip_id"]), "Add a fade in/out to a clip edge."),
    "add_text_layer": (_obj({"track_id": _S(), "text": _S(), "timeline_in": _N(),
                             "timeline_out": _N(), "font_size": _I()},
                            ["track_id", "text", "timeline_in", "timeline_out"]),
                       "Add a burned-in caption/title."),
    "add_audio": (_obj({"asset": _S(), "src_in": _N(), "src_out": _N(),
                        "timeline_in": _N(), "volume": _N(), "track_id": _S()}, ["asset"]),
                  "Add an audio clip (music bed / VO)."),
    "checkpoint": (_obj({"label": _S()}, []), "Snapshot current state."),
    "undo": (_obj({}, []), "Revert to the last checkpoint."),
    "render": (_obj({"output": _S(), "run_qc_after": _B(), "dry_run": _B()}, ["output"]),
               "Compile the timeline to video (deterministic) and optionally QC it."),
    "qc": (_obj({"file": _S(), "expected_duration": _N()}, ["file"]),
           "Run deterministic QC on a rendered file."),
    # --- ingest pipeline --- #
    "ingest_asset": (_obj({"asset_id": _S(), "uri": _S(), "index_dir": _S()},
                          ["asset_id", "uri"]),
                     "Ingest a media file: detect scenes, transcribe, analyze, index."),
    "build_narrative": (_obj({"project_id": _S(), "index_dir": _S()}, []),
                        "Build cross-asset narrative index from all ingested assets."),
    "search_index": (_obj({"query": _S(), "index_dir": _S()}, ["query"]),
                     "Search the semantic media index by text."),
    # --- high-level verbs --- #
    "remove_silences": (_obj({"track_id": _S(), "min_silence": _N()}, ["track_id"]),
                        "Remove silent portions from clips on a track."),
    "reframe_to": (_obj({"asset_id": _S(), "target_aspect": _S(enum=["9:16","16:9","1:1","4:3"]),
                         "focus": _S(enum=["center","top","bottom"])}, ["asset_id"]),
                   "Reframe a video asset for a different aspect ratio."),
    "sync_cuts_to_beat": (_obj({"track_id": _S(), "audio_track_id": _S()},
                               ["track_id", "audio_track_id"]),
                          "Snap clip transitions to the nearest beat in the audio track."),
    "generate_captions": (_obj({"track_id": _S(), "style": _S(enum=["default","bold","minimal","social"]),
                                "index_dir": _S()}, ["track_id"]),
                          "Generate captions from the transcript."),
    "auto_ducking": (_obj({"voice_track_id": _S(), "music_track_id": _S(), "duck_level": _N()},
                          ["voice_track_id", "music_track_id"]),
                     "Automatically duck music volume when voice is present."),
    "assemble_from_plan": (_obj({"plan_json": _S()}, ["plan_json"]),
                           "Assemble a cut from a structured edit plan (JSON string)."),
    "make_short": (_obj({"topic": _S(), "duration": _N(), "index_dir": _S()}, ["topic"]),
                   "Create a short clip from source material matching a topic."),
    # --- perception / VLM review --- #
    "vlm_review": (_obj({"video_path": _S(), "brief": _S(), "model": _S()}, ["video_path"]),
                   "Run VLM semantic review on a rendered video."),
    "render_proxy": (_obj({"output": _S(), "width": _I(), "height": _I()}, ["output"]),
                     "Render a fast low-resolution proxy for review."),
    # --- agents --- #
    "run_agent_swarm": (_obj({"brief": _S(), "target_duration": _N(), "style": _S(),
                              "max_iterations": _I(), "index_dir": _S()}, ["brief"]),
                        "Run the multi-agent swarm to edit a video from a brief."),
    # --- OTIO bridge --- #
    "export_otio": (_obj({"output_path": _S()}, ["output_path"]),
                    "Export the project as an OpenTimelineIO (.otio) file."),
    "import_otio": (_obj({"otio_path": _S()}, ["otio_path"]),
                    "Import an OpenTimelineIO file into the current project."),
    # --- render farm --- #
    "render_farm": (_obj({"output": _S(), "n_workers": _I()}, ["output"]),
                    "Render using the distributed render farm."),
    "render_farm_dry_run": (_obj({}, []),
                            "Show what the render farm would do without rendering."),
}


def _tools_list() -> list[dict]:
    out = []
    for name, fn in TOOLS.items():
        schema, desc = TOOL_SCHEMAS[name]
        out.append({"name": name, "description": desc, "inputSchema": schema})
    return out


# --------------------------------------------------------------------------- #
# JSON-RPC plumbing
# --------------------------------------------------------------------------- #
def _result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _handle(msg: dict) -> Optional[dict]:
    method = msg.get("method")
    id_ = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        return _result(id_, {"protocolVersion": PROTOCOL_VERSION,
                             "capabilities": {"tools": {}},
                             "serverInfo": SERVER_INFO})
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {"tools": _tools_list()})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = TOOLS.get(name)
        if fn is None:
            return _error(id_, -32601, f"unknown tool {name!r}")
        try:
            value = fn(**args)
            text = json.dumps(value, ensure_ascii=False, indent=2)
            return _result(id_, {"content": [{"type": "text", "text": text}],
                                "isError": False})
        except VerbError as e:
            text = json.dumps(e.to_dict(), ensure_ascii=False, indent=2)
            return _result(id_, {"content": [{"type": "text", "text": text}],
                                "isError": True})
        except Exception as e:  # structured, instructive failure
            text = json.dumps({"error": str(e),
                               "trace": traceback.format_exc().splitlines()[-3:]},
                              ensure_ascii=False, indent=2)
            return _result(id_, {"content": [{"type": "text", "text": text}],
                                "isError": True})
    if id_ is not None:
        return _error(id_, -32601, f"unknown method {method!r}")
    return None


def serve(stdin=None, stdout=None) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = _handle(msg)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


if __name__ == "__main__":
    serve()
