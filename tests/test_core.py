"""Unit + regression tests for the Cutible core."""

import math
import os
import subprocess

import pytest

from cutible import Project, Editor, VerbError
from cutible.schema import Clip
from cutible.compiler import FFmpegCompiler

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "examples", "assets")


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_clip_duration_and_speed():
    c = Clip(id="c", asset="a", src_in=2.0, src_out=6.0, timeline_in=1.0, speed=2.0)
    assert c.src_duration == 4.0
    assert c.duration == 2.0           # 4s of source at 2x = 2s
    assert c.timeline_out == 3.0


def test_clip_rejects_bad_span():
    with pytest.raises(Exception):
        Clip(id="c", asset="a", src_in=5.0, src_out=5.0)


def test_project_rejects_unknown_asset_ref():
    with pytest.raises(Exception):
        Project(
            id="p",
            tracks=[{"id": "t", "kind": "video",
                     "clips": [{"id": "c", "asset": "missing", "src_out": 3.0}]}],
        )


def test_content_hash_is_stable_and_order_independent():
    p1 = Project(id="p")
    p2 = Project(id="p")
    assert p1.content_hash() == p2.content_hash()


# --------------------------------------------------------------------------- #
# verbs + diffs
# --------------------------------------------------------------------------- #
def _editor():
    ed = Editor(Project(id="p"))
    ed.add_asset("a", "color", color="red")
    ed.add_track("v", "video")
    return ed


def test_add_clip_returns_diff_with_timecodes():
    ed = _editor()
    diff = ed.add_clip("v", "a", src_in=0, src_out=5)
    assert diff.verb == "add_clip"
    assert diff.details["duration"] == 5.0
    assert diff.changed and diff.changed[0].startswith("clip_")


def test_append_stacks_clips_end_to_end():
    ed = _editor()
    ed.add_clip("v", "a", src_in=0, src_out=4)
    d2 = ed.add_clip("v", "a", src_in=0, src_out=3)
    assert d2.details["timeline_in"] == 4.0     # appended after the first
    assert ed.project.duration == 7.0


def test_split_preserves_total_duration():
    ed = _editor()
    ed.add_clip("v", "a", src_in=0, src_out=6, clip_id="c1")
    before = ed.project.duration
    ed.split("c1", 2.0)
    assert ed.project.duration == before
    clips = ed.project.track("v").clips
    assert len(clips) == 2
    assert math.isclose(clips[0].duration + clips[1].duration, before)


def test_ripple_delete_closes_gap():
    ed = _editor()
    ed.add_clip("v", "a", src_in=0, src_out=4, clip_id="c1")
    ed.add_clip("v", "a", src_in=0, src_out=4, clip_id="c2")
    assert ed.project.duration == 8.0
    ed.ripple_delete("c1")
    clips = ed.project.track("v").clips
    assert [c.id for c in clips] == ["c2"]
    assert clips[0].timeline_in == 0.0          # pulled left
    assert ed.project.duration == 4.0


def test_structured_error_on_bad_clip_id():
    ed = _editor()
    with pytest.raises(VerbError) as ei:
        ed.trim("nope", src_in=1)
    assert ei.value.hint                          # instructive
    assert "known_clips" in ei.value.context


def test_undo_restores_prior_state():
    ed = _editor()
    ed.add_clip("v", "a", src_in=0, src_out=4)
    ed.checkpoint("base")
    ed.add_clip("v", "a", src_in=0, src_out=4)
    assert ed.project.duration == 8.0
    ed.undo()
    assert ed.project.duration == 4.0


def test_branch_is_independent():
    ed = _editor()
    ed.add_clip("v", "a", src_in=0, src_out=4)
    alt = ed.branch()
    alt.add_clip("v", "a", src_in=0, src_out=4)
    assert ed.project.duration == 4.0             # original untouched
    assert alt.project.duration == 8.0


# --------------------------------------------------------------------------- #
# compiler (no ffmpeg needed: dry-run)
# --------------------------------------------------------------------------- #
def test_compiler_dry_run_builds_filtergraph():
    ed = _editor()
    ed.add_clip("v", "a", src_in=0, src_out=3)
    out = FFmpegCompiler(ed.project).render("/dev/null", dry_run=True)
    assert out["duration"] == 3.0
    assert "overlay" in out["filter_complex"]
    assert "color=c=red" in out["command"]   # color generator is an input


def test_compiler_rejects_empty_project():
    with pytest.raises(ValueError):
        FFmpegCompiler(Project(id="empty")).build()


# --------------------------------------------------------------------------- #
# golden render + QC (requires ffmpeg + generated assets)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "speaker_a.mp4")),
                    reason="run examples/make_assets.sh first")
def test_golden_render_and_qc(tmp_path):
    from cutible.qc import run_qc
    ed = Editor(Project(id="g", width=640, height=360))
    ed.add_asset("a", "video", uri=os.path.join(ASSETS, "speaker_a.mp4"), duration=10)
    ed.add_track("v", "video")
    ed.add_clip("v", "a", src_in=0, src_out=3)
    out = str(tmp_path / "g.mp4")
    res = FFmpegCompiler(ed.project).render(out)
    assert res["ok"] and os.path.exists(out)
    report = run_qc(out, expected_duration=3.0)
    assert report.passed
    assert report.has_video and report.has_audio
    assert abs(report.duration - 3.0) < 0.3


@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "speaker_a.mp4")),
                    reason="run examples/make_assets.sh first")
def test_render_is_deterministic(tmp_path):
    ed = Editor(Project(id="d", width=640, height=360))
    ed.add_asset("a", "video", uri=os.path.join(ASSETS, "speaker_a.mp4"), duration=10)
    ed.add_track("v", "video")
    ed.add_clip("v", "a", src_in=1, src_out=4)
    o1, o2 = str(tmp_path / "1.mp4"), str(tmp_path / "2.mp4")
    FFmpegCompiler(ed.project).render(o1)
    FFmpegCompiler(ed.project).render(o2)

    def framemd5(p):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
                            "-i", p, "-map", "0:v", "-f", "framemd5", "-"],
                           capture_output=True, text=True)
        return [l for l in r.stdout.splitlines() if not l.startswith("#")]

    assert framemd5(o1) == framemd5(o2)
