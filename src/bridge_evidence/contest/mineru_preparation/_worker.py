"""Private CPU native-DOCX worker. Run only through the subprocess launcher."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from .prepare_document import _write_json, parser_environment, validate_paths


REQUIRED_VERSIONS = {"mineru": "4.0.12", "docvortex": "0.5.16"}


def require_native_configuration(config):
    features = config.llm_aided.features
    if features.title_leveling or features.cross_page_table_cell_merge:
        raise RuntimeError("LLM-aided title leveling and table merge must both be disabled")


def _run(source, output, *, training_root, expected_sha256, character_budget, input_scope="official-training"):
    source, output, root, digest = validate_paths(source, output, training_root, expected_sha256,
        input_scope=input_scope)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(parser_environment(output))
    attempts = []

    def offline(event, arguments):
        network = (event in {"socket.connect", "socket.sendto"}
            and getattr(arguments[0], "family", None) in (socket.AF_INET, socket.AF_INET6))
        if network or event == "socket.getaddrinfo":
            attempts.append({"event": event})
            raise RuntimeError("Network disabled in local native DOCX preparation")
    # Irreversible hook is confined to this dedicated process.
    sys.addaudithook(offline)
    started = time.monotonic()
    try:
        versions = {name: importlib.metadata.version(name) for name in REQUIRED_VERSIONS}
        if versions != REQUIRED_VERSIONS:
            raise RuntimeError("Unvalidated parser versions; require MinerU 4.0.12 / DocVortex 0.5.16")
        from .source_docx import source_projection
        from .native_repair import preserve_native_spans
        from .mineru_materials import prepare
        from mineru.parser import MinerUParser
        from mineru.config import config
        require_native_configuration(config)
        payload, scope = source_projection(source)
        projected = output / "parsed-input.docx"
        projected.write_bytes(payload)
        with preserve_native_spans() as repair:
            parsed = MinerUParser(tier="flash", parse_mode="txt", image_analysis=False).parse(projected)
        require_native_configuration(config)
        raw = parsed.to_dict()
        _write_json(output / "document.json", raw)
        (output / "document.md").write_text(parsed.markdown(), encoding="utf-8")
        materials = prepare(raw, projected, budget=character_budget)
        materials.update(parsed_input_sha256=materials["docx_sha256"], docx_sha256=digest,
            id="sha256:" + digest, source=str(source), source_scope=scope)
        for packet in materials["packets"]:
            packet.update(report_id=materials["id"], docx_sha256=digest)
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source changed during parsing")
        _write_json(output / "materials.json", materials)
        manifest = {"source": str(source), "source_root": str(root), "input_scope": input_scope, "docx_sha256": digest,
            "parsed_input_sha256": materials["parsed_input_sha256"], "scope": scope,
            "elapsed_seconds": time.monotonic() - started, "repair": repair,
            "network_attempts": attempts, "versions": versions,
            "configuration": {"tier": "flash", "parse_mode": "txt", "image_analysis": False,
                "llm_aided": False, "network": "blocked", "device_scope": "native_docx_cpu",
                "character_budget": character_budget},
            "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))},
            "diagnostics": materials["diagnostics"], "status": "prepared"}
        _write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as exc:
        _write_json(output / "failure.json", {"status": "failed", "source": str(source),
            "docx_sha256": digest, "network_attempts": attempts,
            "elapsed_seconds": time.monotonic() - started,
            "error": {"type": type(exc).__name__, "message": str(exc)}, "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--training-root", type=Path)
    roots.add_argument("--source-root", type=Path)
    parser.add_argument("--input-scope", choices=["official-training", "official-test"], default="official-training")
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--character-budget", type=int, default=6000)
    args = parser.parse_args()
    result = _run(args.source, args.output, training_root=args.source_root or args.training_root,
        expected_sha256=args.expected_sha256, character_budget=args.character_budget, input_scope=args.input_scope)
    print(json.dumps({"status": result["status"], "docx_sha256": result["docx_sha256"]}))


if __name__ == "__main__":
    main()
