# Cutible — Agent-Native Montage Engine

> A headless video-editing engine whose primary **operator is an AI agent**, not a
> human with a mouse. The agent reads the project as data, calls editing *verbs*,
> renders deterministically, and inspects the result through a QC loop — then
> iterates.

## Architecture

```
              AGENT-REVISOR (LLM: planning, reasoning, decisions)
                    │
        ┌───────────┼───────────┐
        │ HANDS     │ EYES      │ MEMORY
        ▼           ▼           ▼
   Verb API    Perception    Semantic Media
   (14 low +   Loop          Index
   8 high)     (VLM+QC)     (scenes+transcript+VLM+embeddings)
        │           │           │
        └─────┬─────┘           │
              ▼                 │
     Timeline-as-Data ◄────────┘
     (JSON, diffable, auditable)
              │
              ▼
     ┌─────────────────────┐
     │ Deterministic Render │  ← FFmpeg (Contour A)
     │ Remotion (Contour B) │  ← Motion graphics
     │ Render Farm          │  ← Distributed GPU
     └─────────────────────┘
              │
              ▼
         QC Gate (deterministic + VLM)
              │
              ▼
     Final Video / OTIO → DaVinci/Premiere
```

## What's Implemented

| Plan concept | Module | Status |
|---|---|---|
| §4 Timeline-as-Data | `cutible/schema.py` | ✅ pydantic, 3 zooms, content hash |
| §3.1 Low-level verbs (14) | `cutible/verbs.py` | ✅ diffs, checkpoint/undo/branch |
| §3.1 High-level verbs (8) | `cutible/verbs_high.py` | ✅ remove_silences, reframe, beat-sync, captions, ducking, assemble, make_short |
| §5 Ingest Pipeline | `cutible/ingest/` | ✅ scenes, Whisper, VLM, audio analysis, embeddings |
| §5 Semantic Media Index | `cutible/index/` | ✅ models, store, text/time/speaker/B-roll search |
| §3.2 Perception Loop | `cutible/perception/` | ✅ VLM review + proxy render |
| §7 Multi-agent Swarm | `cutible/agents/` | ✅ Planner, Editor, Sound, QC, Orchestrator |
| §6.1 Contour A (FFmpeg) | `cutible/compiler.py` | ✅ deterministic render |
| §6.1 Contour B (Remotion) | `cutible/remotion/` | ✅ TSX generation, config |
| §9 OTIO Bridge | `cutible/otio_bridge/` | ✅ export/import to DaVinci/Premiere |
| §6.2 Distributed Render Farm | `cutible/render_farm/` | ✅ scheduler, workers, assembly |
| §8.1 MCP Server | `cutible/mcp_server.py` | ✅ 35 tools, JSON-RPC 2.0/stdio |
| §8.2 REST API | `cutible/api/` | ✅ FastAPI, full CRUD |
| §8.3 Python SDK | `cutible/sdk/` | ✅ in-process + HTTP client |
| §8.4 CLI | `cutible/cli.py` | ✅ render/probe/view/qc/ingest/search/agent/export/import/farm |
| §12.3 Tests | `tests/` | ✅ 30+ tests |

## Quick Start

```bash
pip install -e .                    # core
pip install -e ".[api]"             # + REST API (FastAPI/uvicorn)
pip install -e ".[whisper]"         # + Whisper transcription
pip install -e ".[all]"             # everything

# Generate synthetic assets
bash examples/make_assets.sh

# Watch the agent assemble a recap
python examples/agent_recap_demo.py
```

### CLI

```bash
# Render
python -m cutible render project.json -o out.mp4 --qc

# Ingest a video into the semantic index
python -m cutible ingest speaker /path/to/speaker.mp4

# Search the index
python -m cutible search "moment where speaker discusses AI"

# Run the multi-agent swarm
python -m cutible agent "make a 60s recap about AI" --duration 60

# Export/Import OTIO
python -m cutible export project.json --otio output.otio
python -m cutible import output.otio --save imported.json

# Distributed render farm
python -m cutible farm project.json -o out.mp4 --workers 4

# Start REST API
python -m cutible serve-api --port 8000
```

### Python SDK

```python
from cutible.sdk import CutibleClient

# In-process mode
client = CutibleClient()
client.create_project("demo", fps=30, width=1920, height=1080)
client.add_asset("speaker", "video", uri="speaker.mp4", duration=60)
client.add_track("v_main", "video")
client.add_clip("v_main", "speaker", src_in=0, src_out=10)
result = client.render("output.mp4")

# Run the agent swarm
result = client.run_agent("make a 30s recap", target_duration=30)
```

### REST API

```bash
# Start server
python -m cutible serve-api

# Create project
curl -X POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"id": "demo", "fps": 30}'

# Add clip
curl -X POST http://localhost:8000/projects/demo/verbs \
  -H "Content-Type: application/json" \
  -d '{"verb": "add_clip", "args": {"track_id": "v1", "asset": "a", "src_out": 5}}'

# Render
curl -X POST http://localhost:8000/projects/demo/render \
  -H "Content-Type: application/json" \
  -d '{"output": "out.mp4", "run_qc": true}'
```

### MCP Server (primary agent interface)

```bash
python -m cutible.mcp_server   # speaks JSON-RPC 2.0 over stdio
```

35 tools exposed including: `create_project`, `add_clip`, `trim`, `split`,
`ripple_delete`, `add_transition`, `add_text_layer`, `render`, `qc`,
`ingest_asset`, `search_index`, `remove_silences`, `reframe_to`,
`sync_cuts_to_beat`, `generate_captions`, `auto_ducking`, `make_short`,
`vlm_review`, `render_proxy`, `run_agent_swarm`, `export_otio`, `import_otio`,
`render_farm`.

## Project Layout

```
cutible/
  schema.py           Timeline-as-Data models + zoom views + content hash
  verbs.py            Editor: low-level verbs (14 primitives)
  verbs_high.py       High-level composite verbs (8 intentions)
  compiler.py         Timeline → FFmpeg filtergraph → mp4
  qc.py               Deterministic QC (duration / black frames / LUFS)
  cli.py              Headless CLI (12 commands)
  mcp_server.py       MCP stdio server (35 tools)
  ingest/
    pipeline.py       Ingest orchestrator
    scenes.py         Scene/shot detection (ffmpeg)
    audio_transcribe.py  Whisper transcription + diarization
    vlm.py            VLM visual analysis (Gemini/OpenAI)
    audio_analysis.py  Beat/silence/tempo detection (librosa/ffmpeg)
    embeddings.py     Embedding generation (CLIP/OpenAI)
  index/
    models.py         Semantic index data models
    store.py          Index persistence
    search.py         Text/time/speaker/B-roll search
  perception/
    vlm_review.py     VLM semantic review of renders
    proxy_render.py   Fast low-res proxy renderer
  agents/
    base.py           Base agent + message types
    planner.py        Director/Planner agent
    editor.py         Editor/Montageur agent
    sound.py          Sound Engineer agent
    qc_agent.py       QC/Reviewer agent
    orchestrator.py   Multi-agent swarm coordinator
  remotion/
    compiler.py       Timeline → Remotion (React) project
  otio_bridge/
    exporter.py       Cutible → OpenTimelineIO
    importer.py       OpenTimelineIO → Cutible
  render_farm/
    worker.py         Segment render worker
    scheduler.py      Task scheduler
    manager.py        Distributed render farm manager
  api/
    app.py            FastAPI REST application
  sdk/
    client.py         Python SDK client (in-process + HTTP)
tests/
  test_core.py        Original 15 tests
  test_new_modules.py 20+ tests for new modules
examples/
  agent_recap_demo.py End-to-end agent demo
  make_assets.sh      Synthetic asset generator
```

## Design Principles (Agent-Native)

1. **State is data, not pixels.** The agent reads/diffs/mutates a JSON timeline.
2. **Verbs return diffs.** Each call reports what changed.
3. **Errors teach.** Structured errors with `hint` and `context`.
4. **Try / inspect / revert.** Checkpoint/undo/branch for exploration.
5. **Deterministic render.** Same project → identical frames.
6. **Closed perception loop.** QC gate + VLM review → self-correction.
7. **Semantic memory.** Ingest → indexed content the agent can search.
8. **Multi-agent swarm.** Specialized roles: plan → edit → sound → QC → iterate.
9. **Industry bridge.** OTIO export → DaVinci/Premiere for human finishing.
