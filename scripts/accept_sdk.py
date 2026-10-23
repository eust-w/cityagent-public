#!/usr/bin/env python3
"""Run a real four-stage SDK + model + DOCX integration and persist version evidence."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bridge_evidence.contest.backends import APIBackend
from bridge_evidence.contest.docx_writer import write_extraction_docx
from bridge_evidence.contest.inference_core import infer_document
from bridge_evidence.contest.run_state import atomic_json, file_digest
from bridge_evidence.contest.sdk_acceptance import runtime_identity, validate_acceptance
from bridge_evidence.contest.text_extract import extract_docx_text
from bridge_evidence.contest.tong_workflow import execute_workflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    backend = APIBackend(args.url, args.model, tokenizer)
    backend.probe()
    target = args.output_dir / args.input.name

    def prepare(v):
        return {**v, "text": extract_docx_text(str(args.input))}

    def infer(v):
        return {
            **v,
            **infer_document(
                v["text"], tokenizer, backend, max_model_len=65536, max_new_tokens=4096
            ),
        }

    def validate(v):
        if not v["model_calls"] or v["model_calls"][-1]["status"] != "success":
            raise ValueError("incomplete model")
        return v

    def render(v):
        write_extraction_docx(v["data"], str(target))
        return v

    result = execute_workflow(
        {}, dict(prepare=prepare, infer=infer, validate=validate, render=render)
    )
    proof = {
        "schema_version": 1,
        "runtime": runtime_identity(),
        "workflow": result["workflow"],
        "model_calls": result["payload"]["model_calls"],
        "input": {"path": str(args.input.resolve()), "sha256": file_digest(args.input)},
        "output": {"path": str(target.resolve()), "sha256": file_digest(target)},
    }
    path = args.output_dir / "acceptance.json"
    atomic_json(path, proof)
    validate_acceptance(path)
    print("Real SDK acceptance evidence saved:", path)


if __name__ == "__main__":
    main()
