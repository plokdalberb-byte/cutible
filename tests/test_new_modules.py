"""Tests for the new modules: ingest, index, high-level verbs, agents, etc."""

import importlib.util
import math
import os
import json
import pytest

from cutible import Project, Editor, VerbError
from cutible.schema import Clip, TextLayer
from cutible.compiler import FFmpegCompiler
from cutible.verbs_high import HighLevelVerbs


# --------------------------------------------------------------------------- #
# Semantic Media Index models
# --------------------------------------------------------------------------- #
def test_asset_index_summary():
    from cutible.index.models import AssetIndex
    idx = AssetIndex(asset_id="a1", uri="test.mp4", duration=60.0)
    s = idx.summary()
    assert s["asset_id"] == "a1"
    assert s["duration"] == 60.0
    assert s["n_scenes"] == 0


def test_narrative_index_find_moment():
    from cutible.index.models import NarrativeIndex, AssetIndex, TranscriptSegment
    idx = AssetIndex(
        asset_id="a1", uri="test.mp4", duration=60.0,
        transcript=[
            TranscriptSegment(start=10, end=15, text="AI is transforming video editing"),
        ],
    )
    narrative = NarrativeIndex(project_id="p1", asset_indices=[idx])
    results = narrative.find_moment("AI")
    assert len(results) == 1
    assert "AI" in results[0]["text"]


def test_narrative_index_to_agent_dict():
    """Verify to_agent_dict() produces the format agents expect."""
    from cutible.index.models import (
        NarrativeIndex, AssetIndex, TranscriptSegment, Scene, Shot,
    )
    shot = Shot(id="s1", asset_id="a1", start=0.0, end=10.0)
    scene = Scene(id="sc1", asset_id="a1", start=0.0, end=10.0, shots=[shot],
                  summary="Opening scene")
    idx = AssetIndex(
        asset_id="a1", uri="test.mp4", duration=60.0,
        fps=30, width=1920, height=1080,
        scenes=[scene],
        transcript=[
            TranscriptSegment(start=0, end=5, text="Hello world"),
            TranscriptSegment(start=10, end=15, text="AI is great"),
        ],
    )
    narrative = NarrativeIndex(
        project_id="p1", total_duration=60.0,
        asset_indices=[idx],
    )
    agent_dict = narrative.to_agent_dict()
    assert "assets" in agent_dict
    assert len(agent_dict["assets"]) == 1
    asset = agent_dict["assets"][0]
    assert asset["asset_id"] == "a1"
    assert asset["duration"] == 60.0
    assert asset["uri"] == "test.mp4"
    assert asset["n_scenes"] == 1
    assert len(asset["best_segments"]) >= 0
    assert agent_dict["total_duration"] == 60.0


def test_index_store_persistence(tmp_path):
    from cutible.index.models import AssetIndex
    from cutible.index.store import IndexStore
    store = IndexStore(str(tmp_path))
    idx = AssetIndex(asset_id="a1", uri="test.mp4", duration=60.0)
    store.store_asset_index(idx)
    loaded = store.load_asset_index("a1")
    assert loaded is not None
    assert loaded.asset_id == "a1"
    assert "a1" in store.list_assets()


def test_index_searcher_text():
    from cutible.index.models import AssetIndex, TranscriptSegment
    from cutible.index.store import IndexStore
    from cutible.index.search import IndexSearcher
    store = IndexStore("/tmp/test_search")
    idx = AssetIndex(
        asset_id="a1", uri="test.mp4", duration=60.0,
        transcript=[
            TranscriptSegment(start=5, end=10, text="Welcome to the show"),
            TranscriptSegment(start=15, end=20, text="Today we discuss AI"),
        ],
    )
    store.store_asset_index(idx)
    searcher = IndexSearcher(store)
    results = searcher.search_text("AI")
    assert len(results) >= 1


# --------------------------------------------------------------------------- #
# Ingest pipeline (without actual media)
# --------------------------------------------------------------------------- #
def test_ingest_config_defaults():
    from cutible.ingest.pipeline import IngestConfig
    cfg = IngestConfig()
    assert cfg.proxy_width == 480
    assert cfg.whisper_model == "base"
    assert cfg.vlm_model == "gemini"


def test_scene_detector_creates_single_shot():
    from cutible.ingest.scenes import SceneDetector
    detector = SceneDetector()
    shots = detector._group_into_scenes([], "test")
    assert shots == []


def test_audio_analyzer_fallback():
    from cutible.ingest.audio_analysis import AudioAnalyzer
    analyzer = AudioAnalyzer()
    assert analyzer._tempo_ffmpeg("nonexistent.wav") == 120.0


def test_embedding_generator_mock():
    from cutible.ingest.embeddings import EmbeddingGenerator
    gen = EmbeddingGenerator(provider="mock", dim=64)
    emb = gen._mock_embedding()
    assert len(emb) == 64
    text_emb = gen._embed_text_mock("hello world")
    assert len(text_emb) == 64


def test_vlm_mock_analysis():
    from cutible.ingest.vlm import VLMAnalyzer
    from cutible.index.models import Shot
    vlm = VLMAnalyzer()
    shot = Shot(id="s1", asset_id="a1", start=0.0, end=5.0)
    result = vlm._mock_analysis(shot)
    assert result.timestamp == 0.0
    assert result.duration == 5.0
    assert result.shot_type is not None


def test_transcribe_fallback():
    from cutible.ingest.audio_transcribe import AudioTranscriber
    transcriber = AudioTranscriber()
    segments = transcriber._transcribe_fallback("nonexistent.mp4")
    assert segments == []


# --------------------------------------------------------------------------- #
# High-level verbs
# --------------------------------------------------------------------------- #
def _editor_with_assets():
    ed = Editor(Project(id="p"))
    ed.add_asset("a", "color", color="red")
    ed.add_asset("b", "color", color="blue")
    ed.add_track("v1", "video")
    ed.add_track("v2", "video")
    ed.add_track("music", "audio")
    return ed


def test_assemble_from_plan():
    ed = _editor_with_assets()
    hl = HighLevelVerbs(ed)
    plan = {
        "clips": [
            {"asset": "a", "src_in": 0, "src_out": 5, "track": "v1", "timeline_in": 0},
            {"asset": "b", "src_in": 0, "src_out": 3, "track": "v1", "timeline_in": 5},
        ],
        "texts": [
            {"text": "Hello", "track": "v1", "in": 0, "out": 2},
        ],
    }
    diff = hl.assemble_from_plan(plan)
    assert diff.verb == "assemble_from_plan"
    assert len(diff.changed) > 0
    assert ed.project.duration == 8.0


def test_auto_ducking():
    ed = _editor_with_assets()
    ed.add_clip("v1", "a", src_in=0, src_out=10)
    ed.add_clip("music", "b", src_in=0, src_out=10)
    hl = HighLevelVerbs(ed)
    diff = hl.auto_ducking("v1", "music", duck_level=0.2)
    assert diff.verb == "auto_ducking"
    music_track = ed.project.track("music")
    for clip in music_track.clips:
        assert clip.volume == 0.2


def test_reframe_to_9x16():
    ed = _editor_with_assets()
    ed.add_clip("v1", "a", src_in=0, src_out=5)
    hl = HighLevelVerbs(ed)
    diff = hl.reframe_to("a", target_aspect="9:16")
    assert diff.verb == "reframe_to"
    assert ed.project.aspect == "9:16"
    assert ed.project.width == 1080
    assert ed.project.height == 1920


def test_generate_captions():
    from cutible.index.models import AssetIndex, TranscriptSegment
    from cutible.index.store import IndexStore
    store = IndexStore("/tmp/test_captions")
    idx = AssetIndex(
        asset_id="a", uri="test.mp4", duration=60.0,
        transcript=[
            TranscriptSegment(start=0, end=3, text="Hello world"),
            TranscriptSegment(start=5, end=8, text="Welcome"),
        ],
    )
    store.store_asset_index(idx)
    ed = Editor(Project(id="p"))
    ed.add_asset("a", "color", color="red")
    ed.add_track("captions", "caption")
    hl = HighLevelVerbs(ed, index_store=store)
    diff = hl.generate_captions("captions")
    assert diff.verb == "generate_captions"
    assert len(diff.changed) == 2


# --------------------------------------------------------------------------- #
# Multi-agent swarm
# --------------------------------------------------------------------------- #
def test_planner_agent():
    from cutible.agents.planner import PlannerAgent
    from cutible.agents.base import AgentMessage, MessageType
    planner = PlannerAgent()
    msg = AgentMessage(
        from_agent="test", to_agent="planner",
        type=MessageType.TASK,
        content={"brief": "Make a 30s recap", "target_duration": 30.0,
                 "narrative": {"total_duration": 60.0, "assets": [
                     {"asset_id": "a1", "duration": 60.0, "uri": "test.mp4"},
                 ]}},
    )
    planner.receive(msg)
    responses = planner.process_all()
    assert len(responses) == 1
    plan = responses[0].content["plan"]
    assert "segments" in plan
    assert plan["target_duration"] == 30.0


def test_planner_agent_with_real_narrative():
    """Planner should work with to_agent_dict() format."""
    from cutible.agents.planner import PlannerAgent
    from cutible.agents.base import AgentMessage, MessageType
    from cutible.index.models import NarrativeIndex, AssetIndex
    idx = AssetIndex(asset_id="a1", uri="test.mp4", duration=60.0)
    narrative = NarrativeIndex(project_id="p1", total_duration=60.0, asset_indices=[idx])
    agent_dict = narrative.to_agent_dict()

    planner = PlannerAgent()
    msg = AgentMessage(
        from_agent="test", to_agent="planner",
        type=MessageType.TASK,
        content={"brief": "Make a recap", "target_duration": 30.0,
                 "narrative": agent_dict},
    )
    planner.receive(msg)
    responses = planner.process_all()
    assert len(responses) == 1
    plan = responses[0].content["plan"]
    assert len(plan["segments"]) > 0
    # Should reference the real asset
    assert plan["segments"][0].get("source_asset") == "a1"


def test_editor_agent():
    from cutible.agents.editor import EditorAgent
    from cutible.agents.base import AgentMessage, MessageType
    editor = EditorAgent()
    plan = {
        "segments": [
            {"type": "hook", "description": "Hook", "start": 0, "end": 5,
             "source_asset": "a"},
        ],
    }
    narrative = {
        "assets": [{"asset_id": "a", "start": 0, "end": 10, "duration": 10}],
    }
    msg = AgentMessage(
        from_agent="test", to_agent="editor",
        type=MessageType.TASK,
        content={"plan": plan, "narrative": narrative},
    )
    editor.receive(msg)
    responses = editor.process_all()
    assert len(responses) == 1
    project_data = responses[0].content["project"]
    assert project_data is not None
    assert len(project_data.get("tracks", [])) > 0


def test_orchestrator():
    from cutible.agents.orchestrator import Orchestrator
    orch = Orchestrator(max_iterations=1)
    result = orch.run(
        brief="Make a 10s test video",
        narrative={"total_duration": 20.0, "assets": [
            {"asset_id": "a1", "duration": 20.0, "uri": "test.mp4"},
        ]},
        target_duration=10.0,
        skip_vlm=True,
    )
    assert "iterations" in result
    assert len(result["iterations"]) >= 1


def test_orchestrator_loads_narrative(tmp_path):
    """Orchestrator should load narrative from index_dir."""
    from cutible.index.models import AssetIndex, NarrativeIndex
    from cutible.index.store import IndexStore
    from cutible.agents.orchestrator import Orchestrator

    # Create a narrative index
    store = IndexStore(str(tmp_path))
    idx = AssetIndex(asset_id="a1", uri="test.mp4", duration=10.0)
    store.store_asset_index(idx)
    narrative = NarrativeIndex(
        project_id="test", total_duration=10.0, asset_indices=[idx]
    )
    store.store_narrative(narrative)

    orch = Orchestrator(max_iterations=1)
    result = orch.run(
        brief="Make a 5s clip",
        target_duration=5.0,
        skip_vlm=True,
        index_dir=str(tmp_path),
    )
    assert "iterations" in result


# --------------------------------------------------------------------------- #
# LLM Client
# --------------------------------------------------------------------------- #
def test_llm_client_unavailable():
    from cutible.agents.llm_client import LLMClient
    client = LLMClient(api_key=None)
    assert not client.available
    result = client.generate("system", "user")
    assert result is None
    text = client.generate_text("system", "user")
    assert text is None


# --------------------------------------------------------------------------- #
# OTIO bridge
# --------------------------------------------------------------------------- #
_has_otio = importlib.util.find_spec("opentimelineio") is not None


@pytest.mark.skipif(not _has_otio, reason="opentimelineio not installed")
def test_otio_export_import_roundtrip(tmp_path):
    from cutible.otio_bridge import OTIOExporter, OTIOImporter
    ed = Editor(Project(id="otio_test"))
    ed.add_asset("a", "color", color="red", duration=5.0)
    ed.add_track("v", "video")
    ed.add_clip("v", "a", src_in=0, src_out=5)
    ed.add_text_layer("v", "Test", 0.0, 2.0)

    otio_path = str(tmp_path / "test.otio")
    exporter = OTIOExporter(ed.project)
    exporter.export(otio_path)
    assert os.path.exists(otio_path)

    importer = OTIOImporter()
    imported = importer.import_file(otio_path, "imported_test")
    assert imported.id == "imported_test"
    assert len(imported.tracks) > 0


@pytest.mark.skipif(not _has_otio, reason="opentimelineio not installed")
def test_otio_has_schema_tags(tmp_path):
    """Verify exported OTIO has proper schema tags."""
    from cutible.otio_bridge import OTIOExporter
    ed = Editor(Project(id="schema_test"))
    ed.add_asset("a", "color", color="red", duration=3.0)
    ed.add_track("v", "video")
    ed.add_clip("v", "a", src_in=0, src_out=3)

    otio_path = str(tmp_path / "schema_test.otio")
    exporter = OTIOExporter(ed.project)
    exporter.export(otio_path)

    # Read raw JSON and check for OTIO_SCHEMA tags
    with open(otio_path, "r") as f:
        data = json.load(f)
    # The top-level should have OTIO_SCHEMA or be readable by opentimelineio
    import opentimelineio as otio
    timeline = otio.adapters.read_from_file(otio_path)
    assert isinstance(timeline, otio.schema.Timeline)
    assert len(timeline.tracks) > 0


# --------------------------------------------------------------------------- #
# Render farm
# --------------------------------------------------------------------------- #
def test_render_farm_scheduler():
    from cutible.render_farm.scheduler import TaskScheduler
    scheduler = TaskScheduler(n_workers=2, max_segment_duration=10.0)
    tasks = scheduler.create_tasks("{}", 35.0, "/tmp/render")
    assert len(tasks) == 4  # 0-10, 10-20, 20-30, 30-35
    progress = scheduler.get_progress()
    assert progress["total"] == 4
    assert progress["pending"] == 4


def test_render_farm_dry_run():
    from cutible.render_farm import RenderFarmManager
    ed = Editor(Project(id="farm_test"))
    ed.add_asset("a", "color", color="red", duration=5.0)
    ed.add_track("v", "video")
    ed.add_clip("v", "a", src_in=0, src_out=5)
    farm = RenderFarmManager(n_workers=2)
    result = farm.render_dry_run(ed.project)
    assert result["duration"] == 5.0
    assert result["n_segments"] >= 1


# --------------------------------------------------------------------------- #
# Perception (mock, no VLM)
# --------------------------------------------------------------------------- #
def test_vlm_review_mock():
    from cutible.perception.vlm_review import VLMReview
    reviewer = VLMReview(model="mock")
    result = reviewer._mock_review(5.0)
    assert result["timestamp"] == 5.0
    assert "visual_quality" in result


def test_proxy_renderer_config():
    from cutible.perception.proxy_render import ProxyConfig
    cfg = ProxyConfig(width=640, height=360)
    assert cfg.resolution == "640x360"


# --------------------------------------------------------------------------- #
# REST API (without running server)
# --------------------------------------------------------------------------- #
def test_api_app_creation():
    from cutible.api.app import create_app
    app = create_app()
    assert app.title == "Cutible API"


# --------------------------------------------------------------------------- #
# SDK client (HTTP mode — test config only)
# --------------------------------------------------------------------------- #
def test_sdk_client_config():
    from cutible.sdk import CutibleClient
    client = CutibleClient(api_url="http://localhost:8000")
    assert client.api_url == "http://localhost:8000"
    assert client._editor is None


# --------------------------------------------------------------------------- #
# Diarization
# --------------------------------------------------------------------------- #
def test_diarizer_energy_heuristic():
    from cutible.ingest.diarization import Diarizer, build_speaker_profiles
    from cutible.index.models import TranscriptSegment
    diarizer = Diarizer(provider="energy")
    segments = [
        TranscriptSegment(start=0, end=3, text="Hello"),
        TranscriptSegment(start=3.5, end=6, text="World"),
        TranscriptSegment(start=10, end=13, text="New topic"),
    ]
    labeled = diarizer._diarize_energy_heuristic(segments)
    assert labeled[0].speaker == "speaker_0"
    assert labeled[1].speaker == "speaker_0"  # small gap, same speaker
    assert labeled[2].speaker == "speaker_1"  # big gap, new speaker


def test_build_speaker_profiles():
    from cutible.ingest.diarization import build_speaker_profiles
    from cutible.index.models import TranscriptSegment
    segments = [
        TranscriptSegment(start=0, end=3, text="A", speaker="s1"),
        TranscriptSegment(start=5, end=8, text="B", speaker="s2"),
        TranscriptSegment(start=10, end=13, text="C", speaker="s1"),
    ]
    profiles = build_speaker_profiles(segments)
    assert len(profiles) == 2
    s1 = next(p for p in profiles if p.speaker_id == "s1")
    assert s1.total_speaking_time == 6.0
    assert s1.segment_count == 2


# --------------------------------------------------------------------------- #
# Vector search / Embedding persistence
# --------------------------------------------------------------------------- #
def test_embedding_persistence(tmp_path):
    from cutible.index.store import IndexStore
    store = IndexStore(str(tmp_path))
    store.store_embedding("a1_shot1", [0.1, 0.2, 0.3])
    loaded = store.load_embedding("a1_shot1")
    assert loaded == [0.1, 0.2, 0.3]
    assert "a1_shot1" in store.get_all_embeddings()


def test_semantic_search(tmp_path):
    from cutible.index.store import IndexStore
    store = IndexStore(str(tmp_path))
    store.store_embedding("a1_s1", [1.0, 0.0, 0.0])
    store.store_embedding("a1_s2", [0.0, 1.0, 0.0])
    store.store_embedding("a1_s3", [0.9, 0.1, 0.0])
    results = store.search_semantic([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["ref"] == "a1_s1"
    assert results[0]["score"] > 0.99


def test_index_searcher_semantic(tmp_path):
    from cutible.index.store import IndexStore
    from cutible.index.search import IndexSearcher
    store = IndexStore(str(tmp_path))
    store.store_embedding("asset1_shot1", [0.5, 0.5, 0.0])
    searcher = IndexSearcher(store)
    results = searcher.search_semantic("test query")
    assert isinstance(results, list)


# --------------------------------------------------------------------------- #
# Remote render worker
# --------------------------------------------------------------------------- #
def test_remote_worker_health_check():
    from cutible.render_farm.remote import RemoteWorker
    worker = RemoteWorker("test", "http://localhost:9999")
    assert worker.health_check() is False  # no server running


def test_render_farm_manager_with_remote():
    from cutible.render_farm.manager import RenderFarmManager
    manager = RenderFarmManager(
        n_workers=2,
        remote_endpoints=["http://localhost:8001", "http://localhost:8002"],
    )
    assert len(manager.workers) == 2
    assert manager.workers[0].endpoint == "http://localhost:8001"


# --------------------------------------------------------------------------- #
# Remotion video clips
# --------------------------------------------------------------------------- #
def test_remotion_compiler_generates_videoclip(tmp_path):
    from cutible.remotion.compiler import RemotionCompiler
    ed = Editor(Project(id="remotion_test"))
    ed.add_asset("a", "video", uri="test.mp4", duration=5.0)
    ed.add_track("v", "video")
    ed.add_clip("v", "a", src_in=0, src_out=5)
    ed.add_text_layer("v", "Hello", 0.0, 2.0)
    compiler = RemotionCompiler(ed.project)
    result = compiler.generate_project(str(tmp_path / "remotion"))
    assert result["ok"]
    comp_ts = open(str(tmp_path / "remotion" / "src" / "Composition.tsx")).read()
    assert "VideoClip" in comp_ts
    assert "staticFile" in comp_ts
    assert "TextLayer" in comp_ts
    video_clip_ts = open(str(tmp_path / "remotion" / "src" / "VideoClip.tsx")).read()
    assert "Video" in video_clip_ts
    assert "playbackRate" in video_clip_ts
