"""FastAPI application for the Cutible REST API.

Provides HTTP endpoints for:
- Project CRUD
- Asset management
- Editing operations (verbs)
- Rendering
- QC
- Ingest pipeline
- Agent orchestration
- Semantic index
"""

from __future__ import annotations

import os

try:
    from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..compiler import FFmpegCompiler
from ..schema import Project
from ..verbs import Editor

if HAS_FASTAPI:

    class ProjectCreate(BaseModel):
        id: str
        fps: int = 30
        width: int = 1920
        height: int = 1080
        prompt: str = ""

    class VerbRequest(BaseModel):
        verb: str
        args: dict = {}

    class IngestRequest(BaseModel):
        asset_id: str
        uri: str

    class RenderRequest(BaseModel):
        output: str = "output.mp4"
        run_qc: bool = True

    class AgentRequest(BaseModel):
        brief: str
        target_duration: float = 60.0
        style: str = "informative"
        max_iterations: int = 3
        index_dir: str = ".cutible/index"


def _verify_api_key(request: Request) -> bool:
    """Verify API key if CUTIBLE_API_KEY is set. Open if not set."""
    required_key = os.environ.get("CUTIBLE_API_KEY", "")
    if not required_key:
        return True
    provided = request.headers.get("X-API-Key", "")
    if not provided or provided != required_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Set X-API-Key header.",
        )
    return True


def create_app(title: str = "Cutible API", version: str = "0.1.0") -> FastAPI:
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required: pip install 'cutible[api]'")

    app = FastAPI(title=title, version=version)

    cors_origins = os.environ.get("CUTIBLE_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    project_dir = os.environ.get("CUTIBLE_PROJECT_DIR", ".cutible/projects")
    os.makedirs(project_dir, exist_ok=True)

    def _save_project(project_id: str, editor: Editor) -> None:
        path = os.path.join(project_dir, f"{project_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(editor.project.model_dump_json(indent=2))

    def _load_project(project_id: str) -> Editor | None:
        path = os.path.join(project_dir, f"{project_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                project = Project.model_validate_json(f.read())
            return Editor(project)
        return None

    @app.get("/health")
    def health():
        return {"status": "ok", "version": version}

    @app.post("/projects")
    def create_project(req: ProjectCreate, _: bool = Depends(_verify_api_key)):
        project = Project(
            id=req.id,
            fps=req.fps,
            width=req.width,
            height=req.height,
            provenance={"prompt": req.prompt} if req.prompt else {},
        )
        editor = Editor(project)
        _save_project(req.id, editor)
        return {"created": req.id, "summary": editor.project.summary()}

    @app.post("/projects/{project_id}/load")
    def load_project(project_id: str, body: dict, _: bool = Depends(_verify_api_key)):
        path = body.get("path", "")
        if not path or not os.path.exists(path):
            raise HTTPException(400, f"File not found: {path}")
        project = Project.load(path)
        editor = Editor(project)
        _save_project(project.id, editor)
        return {"loaded": project.id, "summary": project.summary()}

    @app.get("/projects/{project_id}")
    def read_project(project_id: str, zoom: str = "outline", _: bool = Depends(_verify_api_key)):
        editor = _load_project(project_id)
        if not editor:
            raise HTTPException(404, f"Project {project_id!r} not found")
        return editor.project.view(zoom)

    @app.post("/projects/{project_id}/verbs")
    def apply_verb(project_id: str, req: VerbRequest, _: bool = Depends(_verify_api_key)):
        editor = _load_project(project_id)
        if not editor:
            raise HTTPException(404, f"Project {project_id!r} not found")
        try:
            diff = editor.apply(req.verb, **req.args)
            _save_project(project_id, editor)
            return diff.to_dict()
        except Exception as e:
            raise HTTPException(400, detail=str(e)) from e

    @app.post("/projects/{project_id}/render")
    def render_project(
        project_id: str,
        req: RenderRequest,
        background_tasks: BackgroundTasks,
        _: bool = Depends(_verify_api_key),
    ):
        editor = _load_project(project_id)
        if not editor:
            raise HTTPException(404, f"Project {project_id!r} not found")
        FFmpegCompiler(editor.project)
        if req.run_qc:
            from ..qc import run_qc

            report = run_qc(req.output, expected_duration=editor.project.duration)
            return {"output": req.output, "qc": report.to_dict()}
        return {"output": req.output, "duration": editor.project.duration}

    @app.post("/qc")
    def run_qc_endpoint(body: dict, _: bool = Depends(_verify_api_key)):
        from ..qc import run_qc

        report = run_qc(body["file"], expected_duration=body.get("expected_duration"))
        return report.to_dict()

    @app.post("/ingest")
    def ingest_asset(req: IngestRequest, _: bool = Depends(_verify_api_key)):
        from ..ingest import IngestPipeline

        pipeline = IngestPipeline()
        result = pipeline.ingest_asset(req.asset_id, req.uri)
        return result.to_dict()

    @app.post("/agent/run")
    def run_agent(req: AgentRequest, _: bool = Depends(_verify_api_key)):
        import os as _os

        from ..agents.orchestrator import Orchestrator

        openai_key = _os.environ.get("OPENAI_API_KEY")
        openai_base = _os.environ.get("OPENAI_BASE_URL")
        openai_model = _os.environ.get("OPENAI_MODEL")
        orchestrator = Orchestrator(
            max_iterations=req.max_iterations,
            openai_api_key=openai_key,
            openai_base_url=openai_base,
            openai_model=openai_model,
        )
        result = orchestrator.run(
            brief=req.brief,
            target_duration=req.target_duration,
            style=req.style,
            index_dir=req.index_dir,
        )
        return result

    @app.get("/index/{project_id}/narrative")
    def get_narrative(project_id: str, _: bool = Depends(_verify_api_key)):
        from ..index import IndexStore

        index_dir = os.environ.get("CUTIBLE_INDEX_DIR", ".cutible/index")
        store = IndexStore(index_dir)
        narrative = store.load_narrative()
        if narrative is None:
            narrative = store.build_narrative(project_id)
        return narrative.to_agent_dict()

    @app.get("/index/search")
    def search_index(q: str, _: bool = Depends(_verify_api_key)):
        from ..index import IndexSearcher, IndexStore

        index_dir = os.environ.get("CUTIBLE_INDEX_DIR", ".cutible/index")
        store = IndexStore(index_dir)
        searcher = IndexSearcher(store)
        results = searcher.search_text(q)
        return {"results": results, "count": len(results)}

    @app.get("/projects/{project_id}/otio")
    def export_otio(project_id: str, output_path: str, _: bool = Depends(_verify_api_key)):
        from ..otio_bridge import OTIOExporter

        editor = _load_project(project_id)
        if not editor:
            raise HTTPException(404, f"Project {project_id!r} not found")
        exporter = OTIOExporter(editor.project)
        result = exporter.export(output_path)
        return result

    @app.post("/projects/{project_id}/otio/import")
    def import_otio(project_id: str, body: dict, _: bool = Depends(_verify_api_key)):
        from ..otio_bridge import OTIOImporter

        otio_path = body.get("otio_path", "")
        if not otio_path or not os.path.exists(otio_path):
            raise HTTPException(400, f"OTIO file not found: {otio_path}")
        importer = OTIOImporter()
        project = importer.import_file(otio_path, project_id)
        editor = Editor(project)
        _save_project(project_id, editor)
        return {"imported": otio_path, "summary": project.summary()}

    @app.post("/projects/{project_id}/render-farm")
    def render_farm(project_id: str, body: dict, _: bool = Depends(_verify_api_key)):
        from ..render_farm import RenderFarmManager

        editor = _load_project(project_id)
        if not editor:
            raise HTTPException(404, f"Project {project_id!r} not found")
        n_workers = body.get("n_workers", 2)
        output = body.get("output", "output.mp4")
        farm = RenderFarmManager(n_workers=n_workers)
        result = farm.render(editor.project, output)
        return result

    return app
