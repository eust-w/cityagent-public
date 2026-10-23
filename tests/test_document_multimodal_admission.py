"""Subprocess stubs only; no native helper, vLLM, HTTP or model is executed."""
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from bridge_evidence.contest import document_multimodal_admission as m


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def save(path, value):
    path.write_bytes(packed(value) + b"\n")


def payload():
    out = io.BytesIO()
    Image.new("RGB", (4, 3), "blue").save(out, format="PNG")
    return {"model": m.MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "只转写能看清的图片文字。"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()}}
    ]}], "max_tokens": 512, "temperature": 0, "n": 1, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(m, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    base = tmp_path / "model"
    base.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen3_5"}')
    (base / "tokenizer.json").write_text('{"synthetic":true}')
    monkeypatch.setattr(m, "BASE_MODEL", base)
    native_file = tmp_path / "synthetic_native.py"
    native_file.write_text("# Synthetic source fixture; never executed\n")
    argv = [sys.executable, "/synthetic/vllm", "serve", str(base), "--served-model-name", m.MODEL,
            "--max-model-len", "65536", "--dtype", "bfloat16", "--trust-remote-code"]
    audit = {"service_process": {"pid": 12345, "cmdline": argv + [""],
                                "cmdline_sha256": sha(b"\0".join(x.encode() for x in argv) + b"\0")},
             "packages": {"vllm": "synthetic-ppu", "transformers": "synthetic", "pillow": "synthetic"},
             "source_index": [{"path": str(p), "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
                              for p in (native_file, base / "config.json")]}
    audit_path = tmp_path / "audit.json"
    save(audit_path, audit)
    helper = tmp_path / "frozen_helper.py"
    shutil.copyfile(Path(m.__file__).with_name("document_native_processor.py"), helper)
    calls, killed = [], []
    controls = {"mutation": None, "timeout": False, "wait_error": None, "returncode": 0, "request": None}

    class FakeProcess:
        pid = 98765

        def __init__(self, args, **kwargs):
            self.args, self.kwargs, self.waits = args, kwargs, 0
            calls.append({"args": args, **kwargs})
            kwargs["stdout"].write(b"SYNTHETIC PROCESSOR STUB: no native processor executed\n")

        def wait(self, timeout):
            self.waits += 1
            if controls["wait_error"] is not None and self.waits == 1:
                raise controls["wait_error"]
            if controls["timeout"]:
                if self.waits == 1:
                    raise subprocess.TimeoutExpired(self.args, timeout)
                return -9
            if self.waits > 1:
                return controls["returncode"]
            args = self.args
            option = lambda name: Path(args[args.index(name) + 1])
            output, request_path = option("--output"), option("--request")
            output.mkdir()
            request = json.loads(request_path.read_bytes())
            source = json.loads(option("--source").read_bytes())
            copied_audit = json.loads(option("--path-audit").read_bytes())
            original, expanded = list(range(65)), list(range(662))
            summary = {"schema_version": 1, "helper": m.HELPER_VERSION, "status": "processor_counted",
                "method": m.NATIVE_METHOD, "generation_calls": 0, "server_usage_equivalence_verified": False,
                "request_sha256": sha(packed(request)), "messages_sha256": sha(packed(request["messages"])),
                "source_image_sha256": source["blob_sha256"], "image_bytes": source["blob_bytes"],
                "image_dimensions": [source["width"], source["height"]], "max_model_len": 65536,
                "max_output_tokens": 512, "fits_context": True, "processor_apply_calls": 1,
                "bound_input_files": {str(p): sha(p.read_bytes()) for p in
                    (request_path, option("--source"), option("--image"), option("--path-audit"), Path(args[3]))},
                "service_cmdline_sha256": copied_audit["service_process"]["cmdline_sha256"],
                "identity": {"service_pid": copied_audit["service_process"]["pid"],
                    "service_argv": [x for x in copied_audit["service_process"]["cmdline"] if x],
                    "service_cmdline_sha256": copied_audit["service_process"]["cmdline_sha256"],
                    "package_versions": copied_audit["packages"],
                    "verified_audited_files": [{k: row[k] for k in ("path", "sha256", "bytes")} for row in copied_audit["source_index"]],
                    "model_metadata_files": m._metadata()},
                "original_template_prompt_tokens": len(original), "original_template_prompt_token_ids_sha256": sha(packed(original)),
                "expanded_prompt_tokens": len(expanded), "final_prompt_token_ids_sha256": sha(packed(expanded)),
                "image_placeholder_ranges": [{"offset": 4, "length": 598}],
                "native_objects": {key: {"class": ("vllm.model_executor.models.qwen3_vl.Qwen3VLMultiModalProcessor"
                                        if key == "multimodal_processor" else "synthetic." + key),
                                        "source_file": str(native_file), "source_sha256": sha(native_file.read_bytes())}
                    for key in ("renderer", "tokenizer", "input_preprocessor", "multimodal_processor", "hf_processor", "image_processor")},
                "cpu_tensors": [{"path": "mm_kwargs.pixel_values.data", "shape": [2392, 1536],
                                 "dtype": "torch.bfloat16", "device": "cpu"}]}
            save(output / "original-template-prompt-token-ids.json", original)
            save(output / "expanded-prompt-token-ids.json", expanded)
            if controls["mutation"]:
                controls["mutation"](summary, output, request_path)
            save(output / "summary.json", summary)
            return controls["returncode"]

    monkeypatch.setattr(m.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(m.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    environment = {"PATH": "/explicit/bin", "LD_LIBRARY_PATH": "/explicit/ppu/lib:"}
    directory = tmp_path / "not-created-until-call"
    wrapper = m.NativeMultimodalAdmission(directory, processor_python=Path(sys.executable), path_audit=audit_path,
                                         runtime_environment=environment, helper_path=helper)
    return SimpleNamespace(wrapper=wrapper, directory=directory, controls=controls, calls=calls, killed=killed,
                           clock=clock, native_file=native_file, audit_path=audit_path, helper=helper,
                           base=base, audit=audit, environment=environment)


def test_preflight_and_verify_are_readonly_and_identity_is_not_mutable(harness):
    h = harness
    assert not h.directory.exists() and not h.calls
    identity = h.wrapper.identity
    assert identity["service_identity_sha256"] == sha(packed(h.audit["service_process"]))
    identity["model"] = "tampered"
    h.wrapper.verify()
    assert h.wrapper.identity["model"] == m.MODEL
    assert not h.directory.exists() and not h.calls


def test_exact_native_proof_returned_without_token_arrays_and_no_ambient_environment(harness, monkeypatch):
    h = harness
    monkeypatch.setenv("HTTPS_PROXY", "https://do-not-inherit.invalid")
    monkeypatch.setenv("PYTHONPATH", "/do-not-inherit")
    result = h.wrapper(payload(), deadline=100)
    assert result["expanded_prompt_tokens"] == 662 and result["reserved_output_tokens"] == 512
    assert result["method"] == "installed_vllm_native_processor"
    assert result["processor_identity_sha256"] == h.wrapper.identity["processor_identity_sha256"]
    assert result["server_usage_equivalence_verified"] is False and result["generation_calls"] == 0
    assert "prompt_token_ids" not in json.dumps(result) or "final_prompt_token_ids_sha256" in result
    assert len(packed(result)) < 2500
    assert len(h.calls) == 1 and h.calls[0]["start_new_session"] is True
    assert h.calls[0]["env"]["PATH"] == h.environment["PATH"]
    assert h.calls[0]["env"]["LD_LIBRARY_PATH"] == h.environment["LD_LIBRARY_PATH"]
    assert "HTTPS_PROXY" not in h.calls[0]["env"] and "PYTHONPATH" not in h.calls[0]["env"]
    assert h.calls[0]["args"][1:3] == ["-I", "-B"]
    native = Path(result["native_summary"]["path"])
    assert result["native_summary"]["sha256"] == sha(native.read_bytes())
    assert json.loads((native.parent.parent / "summary.json").read_bytes())["status"] == "admitted"
    assert sha(h.helper.read_bytes()) == m.HELPER_SHA256


def test_each_call_has_its_own_artifacts_and_no_hidden_cache(harness):
    h = harness
    first = h.wrapper(payload(), deadline=100)
    second = h.wrapper(payload(), deadline=100)
    assert len(h.calls) == 2 and first["native_summary"]["path"] != second["native_summary"]["path"]
    assert len(list(h.directory.iterdir())) == 2


def test_actual_processor_class_must_be_pinned_before_processing(harness, tmp_path):
    h = harness
    unpinned = tmp_path / "unpinned_native.py"
    unpinned.write_bytes(h.native_file.read_bytes())
    def mutate(summary, output, request):
        summary["native_objects"]["hf_processor"].update(source_file=str(unpinned))
    h.controls["mutation"] = mutate
    with pytest.raises(m.MultimodalAdmissionError, match="MULTIMODAL_NATIVE_OBJECT_INVALID"):
        h.wrapper(payload(), deadline=100)
    assert len(h.calls) == 1
    assert not list(h.directory.glob("*/admission.json"))


@pytest.mark.parametrize("edit", [lambda p: p.update(model="other-model"), lambda p: p.update(max_tokens=511),
    lambda p: p.update(max_tokens=True), lambda p: p.update(temperature=0.1), lambda p: p.update(stream=True),
    lambda p: p.update(tools=[]), lambda p: p["messages"].append(deepcopy(p["messages"][0])),
    lambda p: p["messages"][0]["content"][1]["image_url"].update(url="https://remote/image.png")])
def test_unsupported_requests_do_not_launch_and_leave_failure_receipt(harness, edit):
    h, request = harness, payload()
    edit(request)
    with pytest.raises(m.MultimodalAdmissionError) as caught:
        h.wrapper(request, deadline=100)
    attempt = Path(caught.value.attempt_directory)
    assert not h.calls and json.loads((attempt / "summary.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("edit", [
    lambda s: s.update(status="failed"), lambda s: s.update(method="tokenize_only"),
    lambda s: s.update(generation_calls=True), lambda s: s.update(processor_apply_calls=0),
    lambda s: s.update(request_sha256="0" * 64), lambda s: s.update(source_image_sha256="0" * 64),
    lambda s: s.update(bound_input_files={}), lambda s: s.update(server_usage_equivalence_verified=True),
    lambda s: s["identity"].update(service_pid=67890),
    lambda s: s["identity"].update(package_versions={}),
    lambda s: s["identity"]["verified_audited_files"][0].update(sha256="0" * 64),
    lambda s: s["identity"].update(model_metadata_files=[]),
    lambda s: s.update(expanded_prompt_tokens=65), lambda s: s.update(final_prompt_token_ids_sha256="0" * 64),
    lambda s: s.update(image_placeholder_ranges=[{"offset": 660, "length": 3}]),
    lambda s: s.update(image_placeholder_ranges=[{"offset": True, "length": 3}]),
    lambda s: s["native_objects"]["multimodal_processor"].update(**{"class": "FakeProcessor"}),
    lambda s: s["native_objects"]["hf_processor"].update(source_sha256="0" * 64),
    lambda s: s["cpu_tensors"][0].update(device="cuda:0"),
])
def test_native_summary_claims_require_the_complete_bound_proof(harness, edit):
    h = harness
    h.controls["mutation"] = lambda summary, output, request: edit(summary)
    with pytest.raises(m.MultimodalAdmissionError) as caught:
        h.wrapper(payload(), deadline=100)
    attempt = Path(caught.value.attempt_directory)
    assert len(h.calls) == 1 and not (attempt / "admission.json").exists()
    assert (attempt / "processor.log").is_file() and (attempt / "native/summary.json").is_file()
    assert json.loads((attempt / "summary.json").read_bytes())["status"] == "failed"


@pytest.mark.parametrize("which", ["token_file", "request_file", "original_payload", "helper", "audit", "source", "metadata"])
def test_mutation_cannot_be_admitted(harness, which):
    h, request = harness, payload()
    def mutate(summary, output, request_path):
        if which == "token_file":
            save(output / "expanded-prompt-token-ids.json", [999])
        elif which == "request_file":
            request_path.write_bytes(b"{}")
        elif which == "original_payload":
            request["messages"][0]["content"][0]["text"] = "changed during count"
        else:
            target = {"helper": h.helper, "audit": h.audit_path, "source": h.native_file,
                      "metadata": h.base / "tokenizer.json"}[which]
            target.write_bytes(target.read_bytes() + b" ")
    h.controls["mutation"] = mutate
    with pytest.raises(m.MultimodalAdmissionError):
        h.wrapper(request, deadline=100)
    assert len(h.calls) == 1


@pytest.mark.parametrize("size,success", [(65024, True), (65025, False)])
def test_exact_context_limit_reserves_all_output(harness, size, success):
    h = harness
    def mutate(summary, output, request):
        tokens = list(range(size))
        save(output / "expanded-prompt-token-ids.json", tokens)
        summary.update(expanded_prompt_tokens=size, final_prompt_token_ids_sha256=sha(packed(tokens)))
    h.controls["mutation"] = mutate
    if success:
        assert h.wrapper(payload(), deadline=100)["expanded_prompt_tokens"] == size
    else:
        with pytest.raises(m.MultimodalAdmissionError, match="EXPANSION"):
            h.wrapper(payload(), deadline=100)


def test_timeout_kills_only_own_helper_group_and_preserves_log(harness):
    h = harness
    h.controls["timeout"] = True
    with pytest.raises(m.MultimodalAdmissionError, match="TIMEOUT") as caught:
        h.wrapper(payload(), deadline=100)
    assert h.killed == [(98765, m.signal.SIGKILL)]
    assert all(pid != h.audit["service_process"]["pid"] for pid, _ in h.killed)
    attempt = Path(caught.value.attempt_directory)
    assert (attempt / "processor.log").read_text().startswith("SYNTHETIC")
    assert json.loads((attempt / "summary.json").read_bytes())["generation_calls"] == 0


def test_expired_request_does_not_start_processor(harness):
    h = harness
    with pytest.raises(m.MultimodalAdmissionError, match="DEADLINE"):
        h.wrapper(payload(), deadline=5)
    assert not h.calls and not h.killed


def test_wait_failure_reaps_only_owned_helper_without_masking_original(harness):
    h = harness
    h.controls["wait_error"] = OSError("synthetic wait failure")
    with pytest.raises(m.MultimodalAdmissionError, match="ADMISSION_FAILED") as caught:
        h.wrapper(payload(), deadline=100)
    assert h.killed == [(98765, m.signal.SIGKILL)]
    record = json.loads((Path(caught.value.attempt_directory) / "summary.json").read_bytes())
    assert record["error_type"] == "OSError" and record["status"] == "failed"


def test_late_native_success_is_still_rejected(harness):
    h = harness
    h.controls["mutation"] = lambda s, o, p: h.clock.update(now=101)
    with pytest.raises(m.MultimodalAdmissionError, match="DEADLINE") as caught:
        h.wrapper(payload(), deadline=100)
    assert json.loads((Path(caught.value.attempt_directory) / "summary.json").read_bytes())["status"] == "failed"


def test_counted_summary_with_nonzero_process_exit_is_rejected(harness):
    h = harness
    h.controls["returncode"] = 1
    with pytest.raises(m.MultimodalAdmissionError, match="PROCESSOR_FAILED") as caught:
        h.wrapper(payload(), deadline=100)
    record = json.loads((Path(caught.value.attempt_directory) / "summary.json").read_bytes())
    assert record["processor_returncode"] == 1 and record["status"] == "failed"


def test_deadline_includes_final_receipt_persistence(harness, monkeypatch):
    h = harness
    real_save = m._save
    def slow_save(path, value):
        real_save(path, value)
        if path.name == "summary.json" and value.get("status") == "admitted":
            h.clock["now"] = 101
    monkeypatch.setattr(m, "_save", slow_save)
    with pytest.raises(m.MultimodalAdmissionError, match="DEADLINE") as caught:
        h.wrapper(payload(), deadline=100)
    record = json.loads((Path(caught.value.attempt_directory) / "summary.json").read_bytes())
    assert record["status"] == "failed" and record["error_code"] == "MULTIMODAL_DEADLINE_EXCEEDED"


def test_verify_detects_changes_without_starting_or_creating_directory(harness):
    h = harness
    (h.base / "tokenizer.json").write_text('{"changed":true}')
    with pytest.raises(m.MultimodalAdmissionError, match="IDENTITY_CHANGED"):
        h.wrapper.verify()
    assert not h.calls and not h.directory.exists()
