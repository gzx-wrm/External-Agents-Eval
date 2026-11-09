"""python -m external_agents_eval.preprocessors.codescout {index,run}."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

from .artifacts import ArtifactStore
from .index import build_index
from .pipeline import ChatClient, Config, Preprocessor


def main():
    parser = argparse.ArgumentParser(description="Standalone CodeScout preprocessor")
    parser.add_argument("mode", choices=["index", "run"])
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--issue", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    options = json.loads(args.config.read_text()) if args.config else {}
    unknown = set(options) - {field.name for field in fields(Config)}
    if unknown:
        parser.error("Unknown CodeScout config keys: " + ", ".join(sorted(unknown)))
    config = Config(**options)
    repository, output = args.repository.resolve(), args.output.resolve()
    if repository == output or repository in output.parents:
        parser.error("Output must be outside the repository to keep indexing immutable")
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if args.mode == "run" and (not args.issue or not key):
        parser.error("Run needs --issue and DEEPSEEK_API_KEY in the process environment")
    index = build_index(repository, max_file_bytes=config.max_file_bytes)
    if args.mode == "index":
        store = ArtifactStore(output)
        outline, coverage = index.outline(config.max_outline_chars)
        store.write_json("repository-graph.json", index.graph())
        store.write_text("repository-outline.txt", outline)
        store.write_json("index-coverage.json", coverage)
        print(json.dumps({"files": len(index.files), "entities": len(index.entities), **coverage}))
    else:
        Preprocessor(config, ChatClient(config, key), output, secrets=(key,)).run(
            args.issue.read_text(), index)
        print(json.dumps(json.loads((output / "result.json").read_text())))


if __name__ == "__main__":
    main()
