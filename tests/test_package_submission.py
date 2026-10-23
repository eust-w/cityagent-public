from __future__ import annotations

import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from docx import Document

from bridge_evidence.contest.run_state import file_digest, json_digest
from bridge_evidence.contest.tong_workflow import execute_workflow

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "package_submission.py"
SPEC = importlib.util.spec_from_file_location("package_submission_for_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PACKAGE_SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE_SUBMISSION)
copy_code = PACKAGE_SUBMISSION.copy_code


def test_copy_code_keeps_legacy_student_model_outside_runtime_package(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for name in PACKAGE_SUBMISSION.RUNTIME_FILES:
        runtime_file = source / name
        runtime_file.parent.mkdir(parents=True, exist_ok=True)
        runtime_file.write_text("runtime fixture", encoding="utf-8")
    model = source / "models" / "document_ai" / "bridge_student.json"
    model.parent.mkdir(parents=True)
    model.write_text('{"model_type":"beta_calibrated_candidate_student"}', encoding="utf-8")

    destination = tmp_path / "package" / "code"
    copy_code(source, destination)

    assert (destination / "run.py").is_file()
    assert not (destination / "models").exists()
    assert model.is_file()


@pytest.fixture
def package_fixture(tmp_path):
    """Internal direct baseline fixture; no fake SDK can be accepted by default."""
    source = tmp_path / "source"
    for name in PACKAGE_SUBMISSION.RUNTIME_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic runtime fixture\n")
    (source / "docs").mkdir()
    (source / "docs" / "design.md").write_text("Synthetic design")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    report_input = inputs / "bridge.doc"
    report_input.write_bytes(b"synthetic input")
    results = tmp_path / "results"
    results.mkdir()
    document = Document()
    for _ in range(3):
        document.add_table(rows=1, cols=1)
    output = results / "bridge.docx"
    document.save(output)
    (results / "infer_time.json").write_text('{"infer_time": 0.5}')
    state = tmp_path / "work" / "records"
    state.mkdir(parents=True)
    identity = {"schema_version": 1, "model_files": {"weights.safetensors": "b" * 64},
                "code_files": {name: file_digest(source / name) for name in PACKAGE_SUBMISSION.RUNTIME_FILES},
                "options": {"engine": "direct"}, "packages": {"tongagents": None}}
    fingerprint = json_digest(identity)
    (state.parent / "run_manifest.json").write_text(json.dumps(
        {"schema_version": 1, "identity": identity, "run_fingerprint": fingerprint}))
    call = {"call_id": "synthetic-call", "status": "success", "prompt_sha256": "a" * 64,
            "output_sha256": "c" * 64, "prompt_tokens": 23, "output_tokens": 45,
            "finish_reason": "stop", "seconds": 0.2, "errors": []}
    workflow = execute_workflow({"model_calls": [call]}, {stage: lambda value: value
        for stage in ("prepare", "infer", "validate", "render")}, engine="direct")["workflow"]
    record = {"schema_version": 1, "document": {"relative_path": "bridge.doc", "sha256": file_digest(report_input)},
              "run_fingerprint": fingerprint, "status": "success", "seconds": 0.5,
              "model_validation": {"valid": True}, "model_calls": [call], "workflow": workflow,
              "output": {"name": output.name, "sha256": file_digest(output)}}
    (state / "content-bound.json").write_text(json.dumps(record))
    return {"design_file": source / "docs" / "design.md", "src_root": source, "input_dir": inputs, "result_dir": results, "state_dir": state,
            "output_dir": tmp_path / "fresh-package", "run_fingerprint": fingerprint}


def assert_no_published_archives(directory):
    assert not list(directory.glob("*.tar.gz"))
    assert not list(directory.glob("*.zip"))
    assert not list(directory.glob("*.pending"))


def test_legacy_baseline_publishes_only_after_audit_and_keeps_evidence_outside(package_fixture):
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert report["passed"] is True, report["errors"]
    assert report["sdk_passed"] is False
    output_dir = package_fixture["output_dir"]
    assert (output_dir / "package_root" / "code" / "run.py").is_file()
    assert (output_dir / "audit.json").is_file()
    tar_path = next(output_dir.glob("*.tar.gz"))
    assert "legacy_baseline" in tar_path.name
    with tarfile.open(tar_path, "r:gz") as handle:
        names = handle.getnames()
        assert {name.split("/")[0] for name in names} == {"code", "design", "result"}
        assert not any("records" in name or "run_manifest" in name for name in names)
        assert "code/src/bridge_evidence/contest/execution_audit.py" in names
    with zipfile.ZipFile(next(output_dir.glob("*.zip"))) as handle:
        assert "result/bridge.docx" in handle.namelist()


def test_default_contest_rejects_direct_and_produces_no_release_archive(package_fixture):
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture)
    assert report["passed"] is False
    assert report["profile"] == "contest-sdk"
    assert report["sdk_passed"] is False
    assert_no_published_archives(package_fixture["output_dir"])


@pytest.mark.parametrize("mutation", ["code", "input", "output", "manifest", "extra-output"])
def test_changed_or_untracked_material_cannot_be_packaged(package_fixture, mutation):
    if mutation == "code":
        (package_fixture["src_root"] / "run.py").write_text("# changed code")
    elif mutation == "input":
        (package_fixture["input_dir"] / "bridge.doc").write_bytes(b"same filename new input")
    elif mutation == "output":
        (package_fixture["result_dir"] / "bridge.docx").write_bytes(b"new output")
    elif mutation == "manifest":
        (package_fixture["state_dir"].parent / "run_manifest.json").unlink()
    else:
        (package_fixture["result_dir"] / "unexpected.txt").write_text("must not be silently omitted")
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert report["passed"] is False
    assert_no_published_archives(package_fixture["output_dir"])


def test_secret_audit_still_blocks_before_archive_creation(package_fixture, monkeypatch):
    import bridge_evidence.audit as audit

    builds = []
    monkeypatch.setattr(PACKAGE_SUBMISSION, "build_tar_gz", lambda *args: builds.append(args))
    monkeypatch.setattr(audit, "audit_secrets", lambda paths: ({"finding_files": ["fixture"]}, ["fixture secret rejected"]))
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert report["passed"] is False
    assert builds == []
    assert_no_published_archives(package_fixture["output_dir"])


def test_hash_matching_invalid_docx_is_rejected(package_fixture):
    output = package_fixture["result_dir"] / "bridge.docx"
    output.write_bytes(b"not a Word document")
    record_path = package_fixture["state_dir"] / "content-bound.json"
    record = json.loads(record_path.read_text())
    record["output"]["sha256"] = file_digest(output)
    record_path.write_text(json.dumps(record))
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert report["execution"]["passed"] is True
    assert report["passed"] is False
    assert report["results"]["invalid"]
    assert_no_published_archives(package_fixture["output_dir"])


def test_archive_structure_failure_is_not_published(package_fixture, monkeypatch):
    import bridge_evidence.audit as audit

    monkeypatch.setattr(audit, "audit_archive", lambda path: ({}, ["fixture invalid archive"]))
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert report["passed"] is False
    assert_no_published_archives(package_fixture["output_dir"])


def test_existing_output_is_never_deleted_or_reused(package_fixture):
    package_fixture["output_dir"].mkdir()
    sentinel = package_fixture["output_dir"] / "keep.txt"
    sentinel.write_text("previous work")
    with pytest.raises(FileExistsError):
        PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert sentinel.read_text() == "previous work"


def test_archive_builders_refuse_overwrite(tmp_path):
    root = tmp_path / "package_root"
    root.mkdir()
    for filename, builder in (("prior.tar.gz", PACKAGE_SUBMISSION.build_tar_gz),
                              ("prior.zip", PACKAGE_SUBMISSION.build_zip)):
        destination = tmp_path / filename
        destination.write_bytes(b"previous archive")
        with pytest.raises(FileExistsError):
            builder(root, destination)
        assert destination.read_bytes() == b"previous archive"


def test_cli_requires_explicit_inputs_and_records(monkeypatch):
    monkeypatch.setattr("sys.argv", ["package_submission.py"])
    with pytest.raises(SystemExit) as failure:
        PACKAGE_SUBMISSION.main()
    assert failure.value.code == 2


@pytest.mark.parametrize("arguments", [["--help"], ["--workflow", "document-agent", "--help"]])
def test_actual_packaged_entrypoint_imports_without_repository_fallback(tmp_path, arguments):
    """Exercise the real staging operation, including nested runtime packages."""
    import os
    import subprocess
    import sys

    repository = Path(__file__).resolve().parents[1]
    code = tmp_path / "isolated-code"
    copy_code(repository, code)
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(code / "run.py"), *arguments],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--fusion-policy" in result.stdout


@pytest.mark.parametrize("module", [
    "document_multiagent_cli", "contest_batch_cli", "mineru_preparation",
    "mineru_preparation._worker",
])
def test_packaged_optional_module_help_does_not_import_repository(tmp_path, module):
    import os
    import subprocess
    import sys

    code = tmp_path / "code"
    copy_code(Path(__file__).resolve().parents[1], code)
    bootstrap = (
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module(sys.argv.pop(1),run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", bootstrap, str(code / "src"),
         "bridge_evidence.contest." + module, "--help"],
        cwd=tmp_path, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_source_package_preserves_project_license_and_notice(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = tmp_path / "code"
    copy_code(root, code)
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        assert (code / name).read_bytes() == (root / name).read_bytes()


def test_packaged_python_has_complete_local_import_closure(tmp_path):
    """Cover lazy imports that --help never reaches, without loading a model."""
    import ast

    code = tmp_path / "code"
    copy_code(Path(__file__).resolve().parents[1], code)
    source_root = code / "src"
    missing = []
    # The shared package initializer exposes an unrelated legacy evaluator.
    # Contest runtime imports only evaluation.generation; shipping that legacy
    # export would recursively copy the research orchestrator and label loader.
    excluded_research_imports = {
        ("src/bridge_evidence/evaluation/__init__.py", "bridge_evidence.evaluation.metrics"),
    }

    def check_module(module, source):
        if module == "bridge_evidence" or module.startswith("bridge_evidence."):
            location = source_root.joinpath(*module.split("."))
            if not location.with_suffix(".py").is_file() and not (location / "__init__.py").is_file():
                reference = (source.relative_to(code).as_posix(), module)
                if reference not in excluded_research_imports:
                    missing.append(reference)

    for source in code.rglob("*.py"):
        package = list(source.relative_to(source_root).parts[:-1]) if source.is_relative_to(source_root) else []
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    check_module(alias.name, source)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                if node.module:
                    check_module(".".join([*prefix, node.module]), source)
                elif node.level:
                    for alias in node.names:
                        check_module(".".join([*prefix, alias.name]), source)
    assert not missing, missing
    # These are loaded by filename/module-name rather than an import statement.
    for path in (
        "scripts/probe_fixed_model_vision.py",
        "src/bridge_evidence/contest/document_native_processor.py",
        "src/bridge_evidence/contest/document_multiagent_worker.py",
        "src/bridge_evidence/contest/mineru_preparation/_worker.py",
        "src/bridge_evidence/contest/rsi_namespace.py",
        "src/bridge_evidence/contest/rsi_worker.py",
    ):
        assert (code / path).is_file(), path


def test_copy_code_excludes_experiment_data_and_refuses_linked_sources(tmp_path):
    source = tmp_path / "source"
    for name in PACKAGE_SUBMISSION.RUNTIME_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic runtime\n")
    for name in ("experiments/private.py", "src/bridge_evidence/contest/report.json",
                 "src/bridge_evidence/contest/__pycache__/module.pyc"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private fixture")
    code = tmp_path / "code"
    copy_code(source, code)
    assert not (code / "experiments").exists()
    assert not (code / "src/bridge_evidence/contest/report.json").exists()
    assert not (code / "src/bridge_evidence/contest/__pycache__").exists()
    secret = tmp_path / "outside.py"
    secret.write_text("private fixture")
    (source / "src/bridge_evidence/contest/linked.py").symlink_to(secret)
    with pytest.raises(ValueError, match="符号链接"):
        copy_code(source, tmp_path / "reject-code")


def test_jsonschema_is_declared_for_project_and_sdk_control_environment():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert "jsonschema>=4.26,<5" in project["project"]["dependencies"]
    assert "jsonschema==4.26.0" in (root / "requirements-contest.txt").read_text().splitlines()


def test_optional_cpu_tokenizer_dependencies_are_declared_for_runtime_and_tests():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    for group in ("contest-cpu", "dev"):
        assert "tokenizers>=0.22,<0.23" in extras[group]
        assert "Jinja2>=3.1.6,<4" in extras[group]


def test_package_accepts_recursive_input_and_ignores_word_lockfile(package_fixture):
    inputs = package_fixture["input_dir"]
    nested = inputs / "2026"
    nested.mkdir()
    (inputs / "bridge.doc").rename(nested / "bridge.doc")
    (nested / "~$bridge.docx").write_text("Word lock")
    record_path = package_fixture["state_dir"] / "content-bound.json"
    record = json.loads(record_path.read_text())
    record["document"]["relative_path"] = "2026/bridge.doc"
    record_path.write_text(json.dumps(record))
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert report["passed"] is True, report["errors"]


@pytest.mark.parametrize("seconds", [True, float("inf")])
def test_infer_time_must_be_finite_and_not_boolean(package_fixture, seconds):
    (package_fixture["result_dir"] / "infer_time.json").write_text(json.dumps({"infer_time": seconds}))
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert not report["passed"]
    assert_no_published_archives(package_fixture["output_dir"])


def test_audit_write_failure_removes_only_this_invocations_published_archives(package_fixture, monkeypatch):
    original_open = Path.open

    def fail_audit_write(path, *args, **kwargs):
        if path.name == "audit.json" and args and args[0] == "x":
            raise OSError("synthetic audit write failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_audit_write)
    with pytest.raises(OSError, match="audit write failure"):
        PACKAGE_SUBMISSION.package_submission(**package_fixture, audit_profile="legacy")
    assert_no_published_archives(package_fixture["output_dir"])
    assert (package_fixture["output_dir"] / "package_root").is_dir()


def test_formal_package_requires_original_batch_strong_evidence_and_actual_ledger(package_fixture):
    report = PACKAGE_SUBMISSION.package_submission(**package_fixture)
    assert not report['passed']
    assert any('fresh original' in error for error in report['execution']['errors'])
    assert any('actual timing ledger' in error for error in report['execution']['errors'])
    assert_no_published_archives(package_fixture['output_dir'])
