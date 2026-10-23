"""Explicit display transforms reach both runtime paths and bind run identity."""

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_chunked_integration import CharTokenizer
from test_visual_supported_views import _document, _transform

import run
from bridge_evidence.contest.candidate import validate_config
from bridge_evidence.contest.run_state import json_digest


def _evaluator():
    spec = importlib.util.spec_from_file_location(
        "visual_display_transform_evaluator",
        Path(run.ROOT) / "scripts/evaluate_multitask.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from bridge_evidence.contest import vision_capability

    args = run.build_parser().parse_args(
        [
            "--workflow",
            "chunked",
            "--backend",
            "vllm-api",
            "--engine",
            "direct",
            "--visual-evidence",
            "same-model",
            "--visual-capability",
            str(tmp_path / "capability.json"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    args.visual_capability.write_text("{}")
    args.model_identity = {
        "manifest": {"served_model": "fixed"},
        "files": {"base": {"config.json": "f" * 64}},
    }
    args.visual_trial_id = "a" * 64
    args.candidate = validate_config(
        {"workflow": "chunked", "input_budget": 3000, "output_budget": 500}
    )
    monkeypatch.setattr(
        vision_capability,
        "validate_vision_capability",
        lambda path, identity: {"probe_sha256": "b" * 64},
    )
    run.configure_visual_runtime(args, args.model_identity)
    return args


def test_both_parsers_default_off_and_accept_explicit_flag():
    assert not run.build_parser().parse_args([]).visual_display_transforms
    assert (
        run.build_parser()
        .parse_args(["--visual-display-transforms"])
        .visual_display_transforms
    )
    base = [
        "--manifest",
        "m.json",
        "--model-manifest",
        "model.json",
        "--candidate-config",
        "c.json",
        "--output-dir",
        "out",
        "--hardware",
        "fixture",
    ]
    parser = _evaluator().build_parser()
    assert not parser.parse_args(base).visual_display_transforms
    assert parser.parse_args(
        base + ["--visual-display-transforms"]
    ).visual_display_transforms


@pytest.mark.parametrize(
    "field,value",
    [
        ("visual_evidence", "off"),
        ("workflow", "whole"),
        ("workflow", "document-agent"),
        ("workflow", "divide-fixed"),
        ("backend", "local-vllm"),
        ("visual_capability", None),
        ("text_dir", Path("external-text")),
        ("visual_display_transforms", "true"),
    ],
)
def test_invalid_display_transform_routes_rejected_before_model_work(
    runtime, field, value
):
    runtime.visual_display_transforms = True
    setattr(runtime, field, value)
    with pytest.raises(ValueError):
        run.validate_visual_options(runtime)


def test_explicit_view_policy_changes_identity_conditions_only_when_enabled(runtime):
    legacy = run.configure_visual_runtime(runtime, runtime.model_identity)
    assert legacy["config"] == {"max_tokens": 2048}
    runtime.visual_display_transforms = True
    enabled = run.configure_visual_runtime(runtime, runtime.model_identity)
    assert enabled["config"] == {"max_tokens": 2048, "allow_display_transforms": True}
    assert json_digest(legacy) != json_digest(enabled)
    assert legacy["capability"] == enabled["capability"]


def test_real_cpu_prepare_applies_only_explicit_transform_policy(runtime, tmp_path):
    path, _ = _document(
        tmp_path, lambda picture, _: _transform(picture, crop={"l": "25000"})
    )
    legacy = run.prepare_document(runtime, path, CharTokenizer())
    assert legacy["visual_plan"]["unresolved"] and not legacy["visual_plan"]["jobs"]
    runtime.visual_display_transforms = True
    enabled = run.prepare_document(runtime, path, CharTokenizer())
    assert (
        not enabled["visual_plan"]["unresolved"]
        and len(enabled["visual_plan"]["jobs"]) == 1
    )
    assert (
        enabled["visual_plan"]["identity"]["config"]["allow_display_transforms"] is True
    )
    assert enabled["snapshot"].digest == legacy["snapshot"].digest
    runtime.visual_evidence = "off"
    runtime.visual_capability = None
    with pytest.raises(ValueError, match="same-model"):
        run.prepare_document(runtime, path, CharTokenizer())


@pytest.mark.parametrize("prepare_enabled", [False, True])
def test_changing_view_policy_after_prepare_rejects_before_visual_request(
    runtime, tmp_path, prepare_enabled
):
    path, _ = _document(
        tmp_path, lambda picture, _: _transform(picture, rotation=5400000)
    )
    runtime.visual_display_transforms = prepare_enabled
    prepared = run.prepare_document(runtime, path, CharTokenizer())
    runtime.visual_display_transforms = not prepare_enabled
    with pytest.raises(ValueError, match="display transform policy changed"):
        run.infer_with_visual_evidence(
            runtime,
            prepared,
            CharTokenizer(),
            lambda *a: pytest.fail("mismatched view policy reached generation"),
            fingerprint=runtime.visual_trial_id,
            document={},
            record={},
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_evaluator_forwards_policy_to_preparation_execution_and_run_identity(
    tmp_path, monkeypatch, runtime, enabled
):
    evaluator = _evaluator()
    gold = {"基本信息": {"桥梁名称": "测试桥"}, "维修建议": [], "病害明细": []}
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold))
    cases = [
        {"id": "test", "report": str(tmp_path / "report.docx"), "gold": str(gold_path)}
    ]
    config = tmp_path / "config.json"
    config.write_text('{"workflow":"chunked"}')
    monkeypatch.setattr(
        evaluator, "load_cases", lambda path: ({"samples": cases}, cases)
    )
    model = SimpleNamespace(
        base=str(tmp_path),
        tokenizer=str(tmp_path),
        served_model="test",
        identity=lambda: runtime.model_identity,
    )
    model.validate = lambda: model
    monkeypatch.setattr(evaluator.ModelManifest, "load", lambda path: model)
    identities, observed = [], []

    def build_identity(base, root, options, **kwargs):
        identities.append(options)
        return {"code_files": {}}

    monkeypatch.setattr(evaluator, "build_run_identity", build_identity)
    monkeypatch.setattr(
        evaluator, "bind_work_directory", lambda *a, **kw: "test-fingerprint"
    )
    monkeypatch.setattr(
        evaluator, "comparison_identity", lambda *a, **kw: "test-identity"
    )
    monkeypatch.setattr(
        evaluator, "APIBackend", lambda *a, **kw: SimpleNamespace(probe=lambda: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(eos_token_id=0)
            )
        ),
    )
    entry = ModuleType("contest_runtime")
    entry.validate_visual_options = run.validate_visual_options
    entry.configure_visual_runtime = run.configure_visual_runtime
    entry.build_parser = run.build_parser

    def prepare(args, path, tokenizer):
        run.validate_visual_options(args)
        observed.append(("prepare", args.visual_display_transforms))
        return {}

    def process(args, *a, **kw):
        run.validate_visual_options(args)
        observed.append(("process", args.visual_display_transforms))
        return {
            "final_data": gold,
            "raw_output": json.dumps(gold),
            "report_wall_seconds": 0.01,
        }

    entry.prepare_document, entry.process_document = prepare, process

    class Loader:
        def create_module(self, spec):
            return entry

        def exec_module(self, module):
            pass

    original_spec = importlib.util.spec_from_file_location
    monkeypatch.setattr(
        evaluator.importlib.util,
        "spec_from_file_location",
        lambda name, path: (
            importlib.machinery.ModuleSpec(name, Loader())
            if name == "contest_runtime"
            else original_spec(name, path)
        ),
    )
    output = tmp_path / "output"
    output.mkdir()
    args = evaluator.build_parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--model-manifest",
            "model.json",
            "--candidate-config",
            str(config),
            "--output-dir",
            str(output),
            "--hardware",
            "fixture",
            "--warmup",
            "0",
            "--visual-evidence",
            "same-model",
            "--visual-capability",
            str(runtime.visual_capability),
            *(["--visual-display-transforms"] if enabled else []),
        ]
    )
    result = evaluator._evaluate_locked(args)
    assert result["completed"] == 1 and result["failures"] == 0
    assert observed == [("prepare", enabled), ("process", enabled)]
    bound = identities[0]["visual_evidence"]["config"]
    assert bound.get("allow_display_transforms", False) is enabled
