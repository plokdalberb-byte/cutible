# Repository Guidelines

## Project Structure & Module Organization

Cutible is a headless video-editing engine with 5 architectural layers:

| Layer | Modules | Role |
|---|---|---|
| L1 Core | `schema.py`, `verbs.py` | Timeline-as-Data (Pydantic) + 14 low-level verb primitives |
| L2 Render | `compiler.py`, `qc.py` | Deterministic FFmpeg render + probe-based QC |
| L3 Capabilities | `verbs_high.py`, `ingest/`, `index/`, `perception/`, `render_farm/`, `remotion/`, `otio_bridge/` | Composite verbs, media ingest, semantic search, VLM review, distributed render, OTIO interop |
| L4 Agents | `agents/` | Multi-agent swarm: Planner → Editor → Sound → QC → Orchestrator |
| L5 Interfaces | `cli.py`, `mcp_server.py`, `api/app.py`, `sdk/client.py`, `sdk_ts/` | CLI, MCP (35 tools), REST API, Python SDK, TypeScript SDK |

The primary agent interface is the **MCP server** (`mcp_server.py`): 35 JSON-RPC 2.0 tools over stdio. The REST API and SDKs are secondary.

Key pattern: every verb returns a `Diff`, errors return `VerbError(hint, context)`. The `Project` (Pydantic model) is the single source of truth — JSON, diffable, auditable.

## Build, Test, and Development Commands

```bash
# Install (from repo root)
pip install -e ".[all,dev]"          # full install + dev tools
pip install -e "."                   # core only
pip install -e ".[api]"              # REST API only

# Lint & type-check
ruff check cutible/                  # lint (E, F, W, I, N, UP, B, SIM rules)
ruff format --check cutible/         # format check
mypy cutible/ --ignore-missing-imports

# Run tests
pytest                              # all tests
pytest tests/test_core.py           # core tests only
pytest -k "test_undo"               # specific test by name

# TypeScript SDK (cd sdk_ts/)
npm test                            # jest tests
npx tsc --noEmit                    # type-check (strict mode)

# Docker
docker-compose up app               # REST API on :8000
docker-compose --profile distributed up worker  # render farm worker

# CLI entry points
cutible                             # main CLI (12 commands)
cutible-mcp                         # MCP stdio server
cutible-api --host 0.0.0.0 --port 8000
```

## Coding Style & Naming Conventions

- **Linter**: ruff (`line-length=100`, target `py310`). Selected rules: `E`, `F`, `W`, `I` (isort), `N` (pep8-naming), `UP` (pyupgrade), `B` (flake8-bugbear), `SIM` (flake8-simplify). `E501` (line length) is ignored — use ruff format instead.
- **Type checking**: mypy (`python_version=3.10`, `warn_return_any=true`, `ignore_missing_imports=true`).
- **TypeScript**: `strict: true` in `sdk_ts/tsconfig.json`.
- **No pre-commit hooks** configured.

## Testing Guidelines

- **Python**: pytest, test files in `tests/`. No fixtures/conftest. Some tests use `pytest.mark.skipif` for tests requiring real media assets (generate with `examples/make_assets.sh`).
- **TypeScript**: jest with `ts-jest`, tests in `sdk_ts/src/__tests__/`, matched by `**/__tests__/**/*.test.ts`.

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `main` with 4 jobs: `lint` (ruff check + format check), `typecheck` (mypy), `test` (pytest), `build` (python -m build).