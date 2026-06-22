"""Demo: an agent assembling a 'recap' through Cutible's verb API.

This is a literal, runnable version of plan Phase 1's flagship scenario —
"long-form -> recap" — but driven entirely by the agent-native HANDS (verbs),
proving the loop: act (verbs) -> read state -> render -> QC (eyes).

Run:
    python examples/agent_recap_demo.py
"""

import json
import os

from cutible import Project, Editor
from cutible.compiler import FFmpegCompiler
from cutible.qc import run_qc

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)


def log(diff):
    print("  diff:", json.dumps(diff.to_dict(), ensure_ascii=False))


def main():
    # --- the agent starts a project --------------------------------------- #
    project = Project(
        id="recap_demo", fps=30, width=1920, height=1080,
        provenance={"agent_run_id": "run_001",
                    "prompt": "Make a ~14s recap of the two interview clips with "
                              "title, captions and a music bed."},
    )
    ed = Editor(project)

    # --- register source material (ingested assets) ----------------------- #
    ed.add_asset("title", "image", uri=os.path.join(ASSETS, "titlecard.png"))
    ed.add_asset("cam_a", "video", uri=os.path.join(ASSETS, "speaker_a.mp4"), duration=10)
    ed.add_asset("cam_b", "video", uri=os.path.join(ASSETS, "speaker_b.mp4"), duration=10)
    ed.add_asset("bed", "audio", uri=os.path.join(ASSETS, "music.m4a"), duration=30)

    # --- the agent builds the cut with low-level verbs -------------------- #
    ed.add_track("v_main", "video")
    ed.add_track("captions", "caption")

    print("add_clip title (3s opener)")
    log(ed.add_clip("v_main", "title", src_in=0, src_out=3, speed=1.0,
                    rationale="branded opener / title card"))

    print("add_clip best bit of speaker A")
    log(ed.add_clip("v_main", "cam_a", src_in=2.0, src_out=7.0,
                    rationale="A's strongest 5s — the hook"))

    print("add_clip speaker B reaction")
    log(ed.add_clip("v_main", "cam_b", src_in=1.0, src_out=7.0,
                    rationale="B's response, keeps momentum"))

    # the agent inspects, then tightens B with a trim + speed-up
    clips = ed.read("outline")["tracks"][0]["clips"]
    b_clip = clips[-1]["id"]
    print(f"trim {b_clip} to 4s and 1.25x to keep energy")
    log(ed.trim(b_clip, src_in=1.0, src_out=6.0))
    log(ed.set_speed(b_clip, 1.25))

    # smooth the seams
    print("add fades on the seams")
    for c in ed.read("outline")["tracks"][0]["clips"]:
        ed.add_transition(c["id"], "in", 0.4)
        ed.add_transition(c["id"], "out", 0.4)

    # --- captions (text-based layer) -------------------------------------- #
    print("add captions")
    total = ed.project.duration
    log(ed.add_text_layer("captions", "CUTIBLE \u00b7 agent-made recap",
                          0.2, 3.0, font_size=72))
    log(ed.add_text_layer("captions", "the hook \u2014 speaker A", 3.2, 8.0))
    log(ed.add_text_layer("captions", "the response \u2014 speaker B", 8.2, total - 0.2))

    # --- music bed under everything (auto-ducked low) --------------------- #
    print("add music bed")
    log(ed.add_audio("bed", src_in=0, src_out=total, timeline_in=0,
                     volume=0.18, track_id="music",
                     rationale="low ambient bed, ducked under dialogue"))

    # --- the agent reads its finished timeline ---------------------------- #
    print("\n=== SUMMARY VIEW (what the agent sees) ===")
    print(json.dumps(ed.read("summary"), indent=2, ensure_ascii=False))

    # save the timeline-as-data
    proj_path = os.path.join(OUT, "recap.json")
    ed.project.save(proj_path)
    print(f"\nsaved timeline -> {proj_path}")
    print("content_hash:", ed.project.content_hash())

    # --- render (the deterministic engine) -------------------------------- #
    print("\n=== RENDER ===")
    comp = FFmpegCompiler(ed.project)
    out_mp4 = os.path.join(OUT, "recap.mp4")
    result = comp.render(out_mp4)
    print(json.dumps(result, indent=2))

    # --- QC (the agent's deterministic eyes) ------------------------------ #
    print("\n=== QC ===")
    report = run_qc(out_mp4, expected_duration=ed.project.duration,
                    loudness_target=ed.project.globals.loudness_target)
    print(json.dumps(report.to_dict(), indent=2))
    print("\nQC PASSED" if report.passed else "\nQC FAILED")


if __name__ == "__main__":
    main()
