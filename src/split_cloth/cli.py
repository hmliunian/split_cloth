from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .dataset import select_pilot
from .pipeline import annotate_run, prepare_run, register_expert, summarize
from .util import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description="Expert-guided clothing stage annotation")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ["select", "prepare", "annotate", "register-expert", "summary", "evaluate-gpt", "compare", "event-probe"]:
        p = subs.add_parser(name)
        p.add_argument("--run", required=True)
        if name == "prepare":
            p.add_argument("--expert-only", action="store_true")
        if name in ("annotate", "event-probe"):
            p.add_argument("--limit", type=int)
        if name == "register-expert":
            p.add_argument("--review", type=Path, required=True)
        if name in ("compare", "event-probe"):
            p.add_argument("--baseline", required=True)
        if name == "event-probe":
            p.add_argument("--prepare-only", action="store_true", help="Prepare event evidence and payload inventory without model calls")
    server = subs.add_parser("serve")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    project = args.project.resolve()
    if args.command == "serve":
        from .web import serve
        serve(project, args.host, args.port)
        return
    config_path = args.config or project / "configs/coarse_v1.yaml"
    config = yaml.safe_load(config_path.read_text())
    run = project / "runs" / args.run
    if args.command == "select":
        result = select_pilot(project, run, config)
        print(json.dumps({"selected": len(result), "run": str(run)}))
    elif args.command == "prepare":
        prepare_run(project, run, config, args.expert_only)
    elif args.command == "register-expert":
        register_expert(project, run, config, args.review)
        print("Expert registered with GPT visual provenance")
    elif args.command == "annotate":
        summary = annotate_run(project, run, config, args.limit)
        print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, ensure_ascii=False))
    elif args.command == "summary":
        print(json.dumps({k: v for k, v in summarize(run).items() if k != "episodes"}, ensure_ascii=False))
    elif args.command == "evaluate-gpt":
        from .evaluation import evaluate_gpt
        print(json.dumps({k: v for k, v in evaluate_gpt(run).items() if k != "episodes"}, ensure_ascii=False))
    elif args.command == "compare":
        from .comparison import compare_runs
        result = compare_runs(project / "runs" / args.baseline, run)
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, ensure_ascii=False))
    elif args.command == "event-probe":
        from .event_probe import run_event_probe
        result = run_event_probe(project, run, project / "runs" / args.baseline, config, args.limit, args.prepare_only)
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
