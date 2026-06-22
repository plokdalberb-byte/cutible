"""Cutible CLI — headless "video as code" (plan §8.4).

python -m cutible render   project.json -o out.mp4
python -m cutible probe    project.json
python -m cutible view     project.json --zoom outline
python -m cutible qc       out.mp4 --expect 12.0
python -m cutible ingest   asset_id /path/to/video.mp4
python -m cutible search   "moment where speaker talks about AI"
python -m cutible agent    "make a 60s recap of this interview"
python -m cutible export   project.json --otio output.otio
python -m cutible import   input.otio --project my_project
python -m cutible farm     project.json -o out.mp4 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys

from .compiler import FFmpegCompiler
from .qc import run_qc
from .schema import Project


def _cmd_render(args) -> int:
    project = Project.load(args.project)
    comp = FFmpegCompiler(project)
    result = comp.render(args.output, quiet=not args.verbose)
    print(json.dumps(result, indent=2))
    if args.qc:
        report = run_qc(
            args.output,
            expected_duration=project.duration,
            loudness_target=project.globals.loudness_target,
        )
        print(json.dumps({"qc": report.to_dict()}, indent=2))
        return 0 if report.passed else 2
    return 0


def _cmd_probe(args) -> int:
    project = Project.load(args.project)
    comp = FFmpegCompiler(project)
    print(json.dumps(comp.render("/dev/null", dry_run=True), indent=2))
    return 0


def _cmd_view(args) -> int:
    project = Project.load(args.project)
    from .verbs import Editor

    print(json.dumps(Editor(project).read(args.zoom), indent=2, ensure_ascii=False))
    return 0


def _cmd_qc(args) -> int:
    report = run_qc(args.file, expected_duration=args.expect, loudness_target=args.loudness)
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 2


def _cmd_ingest(args) -> int:
    from .ingest import IngestPipeline
    from .ingest.pipeline import IngestConfig

    pipeline = IngestPipeline(IngestConfig(index_dir=args.index_dir))
    result = pipeline.ingest_asset(args.asset_id, args.uri)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.success else 1


def _cmd_build_index(args) -> int:
    from .ingest import IngestPipeline
    from .ingest.pipeline import IngestConfig

    pipeline = IngestPipeline(IngestConfig(index_dir=args.index_dir))
    narrative = pipeline.build_narrative(args.project_id)
    print(json.dumps(narrative.summary(), indent=2))
    return 0


def _cmd_search(args) -> int:
    from .index import IndexSearcher, IndexStore

    store = IndexStore(args.index_dir)
    searcher = IndexSearcher(store)
    results = searcher.search_text(args.query)
    print(
        json.dumps(
            {"query": args.query, "n_results": len(results), "results": results[:20]},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _cmd_agent(args) -> int:
    import os

    from .agents.orchestrator import Orchestrator
    from .schema import Project

    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base = os.environ.get("OPENAI_BASE_URL")
    openai_model = os.environ.get("OPENAI_MODEL")
    orchestrator = Orchestrator(
        max_iterations=args.max_iterations,
        openai_api_key=openai_key,
        openai_base_url=openai_base,
        openai_model=openai_model,
    )
    result = orchestrator.run(
        brief=args.brief,
        target_duration=args.duration,
        style=args.style,
        index_dir=args.index_dir,
    )

    # Save project to file for rendering
    project_data = result.get("final_project")
    if project_data:
        out_file = args.output or "project.json"
        project = Project.model_validate(project_data)
        project.save(out_file)
        print(f"Project saved to: {out_file}", file=sys.stderr)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("passed") else 1


def _cmd_export_otio(args) -> int:
    from .otio_bridge import OTIOExporter

    project = Project.load(args.project)
    exporter = OTIOExporter(project)
    result = exporter.export(args.output)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_import_otio(args) -> int:
    from .otio_bridge import OTIOImporter

    importer = OTIOImporter()
    project = importer.import_file(args.otio_path, args.project_id)
    if args.save:
        project.save(args.save)
        print(
            json.dumps(
                {"imported": args.otio_path, "saved": args.save, "summary": project.summary()},
                indent=2,
            )
        )
    else:
        print(json.dumps({"imported": args.otio_path, "summary": project.summary()}, indent=2))
    return 0


def _cmd_farm(args) -> int:
    from .render_farm import RenderFarmManager

    project = Project.load(args.project)
    farm = RenderFarmManager(n_workers=args.workers)
    result = farm.render(project, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_farm_dry_run(args) -> int:
    from .render_farm import RenderFarmManager

    project = Project.load(args.project)
    farm = RenderFarmManager(n_workers=args.workers)
    result = farm.render_dry_run(project)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_serve_api(args) -> int:
    from .api.app import create_app

    try:
        import uvicorn

        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("uvicorn required: pip install 'cutible[api]'", file=sys.stderr)
        return 1
    return 0


def serve_api():
    """Entry point for cutible-api console script."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    import argparse

    p = argparse.ArgumentParser(prog="cutible-api")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    _cmd_serve_api(args)


def main(argv=None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    p = argparse.ArgumentParser(prog="cutible", description="Agent-native montage engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a project to video")
    r.add_argument("project")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--qc", action="store_true", help="run QC after render")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=_cmd_render)

    pr = sub.add_parser("probe", help="dry-run: print ffmpeg command + content hash")
    pr.add_argument("project")
    pr.set_defaults(func=_cmd_probe)

    v = sub.add_parser("view", help="print a zoom view of the project")
    v.add_argument("project")
    v.add_argument("--zoom", default="outline", choices=["summary", "outline", "detail"])
    v.set_defaults(func=_cmd_view)

    q = sub.add_parser("qc", help="run QC on a rendered file")
    q.add_argument("file")
    q.add_argument("--expect", type=float, default=None, help="expected duration (s)")
    q.add_argument("--loudness", type=float, default=-14.0)
    q.set_defaults(func=_cmd_qc)

    ing = sub.add_parser("ingest", help="ingest a media file into the semantic index")
    ing.add_argument("asset_id")
    ing.add_argument("uri")
    ing.add_argument("--index-dir", default=".cutible/index")
    ing.set_defaults(func=_cmd_ingest)

    bld = sub.add_parser("build-index", help="build narrative index from ingested assets")
    bld.add_argument("--project-id", default="default")
    bld.add_argument("--index-dir", default=".cutible/index")
    bld.set_defaults(func=_cmd_build_index)

    sch = sub.add_parser("search", help="search the semantic media index")
    sch.add_argument("query")
    sch.add_argument("--index-dir", default=".cutible/index")
    sch.set_defaults(func=_cmd_search)

    ag = sub.add_parser("agent", help="run the multi-agent swarm to edit a video")
    ag.add_argument("brief")
    ag.add_argument("-o", "--output", default="project.json", help="output project file")
    ag.add_argument("--duration", type=float, default=60.0, help="target duration (s)")
    ag.add_argument(
        "--style",
        default="informative",
        choices=["informative", "energetic", "calm", "professional", "humorous"],
    )
    ag.add_argument("--max-iterations", type=int, default=3)
    ag.add_argument("--index-dir", default=".cutible/index", help="path to semantic media index")
    ag.set_defaults(func=_cmd_agent)

    exp = sub.add_parser("export", help="export project as OpenTimelineIO")
    exp.add_argument("project")
    exp.add_argument("-o", "--output", required=True)
    exp.set_defaults(func=_cmd_export_otio)

    im = sub.add_parser("import", help="import an OpenTimelineIO file")
    im.add_argument("otio_path")
    im.add_argument("--project-id", default=None)
    im.add_argument("--save", default=None, help="save imported project to path")
    im.set_defaults(func=_cmd_import_otio)

    fm = sub.add_parser("farm", help="render using distributed render farm")
    fm.add_argument("project")
    fm.add_argument("-o", "--output", required=True)
    fm.add_argument("--workers", type=int, default=2)
    fm.set_defaults(func=_cmd_farm)

    fd = sub.add_parser("farm-dry-run", help="show render farm plan without rendering")
    fd.add_argument("project")
    fd.add_argument("--workers", type=int, default=2)
    fd.set_defaults(func=_cmd_farm_dry_run)

    api = sub.add_parser("serve-api", help="start the REST API server")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.set_defaults(func=_cmd_serve_api)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
