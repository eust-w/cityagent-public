"""Private converter configuration must not escape its extracted package tree."""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/configure_private_libreoffice.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("private_libreoffice", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installation(tmp_path):
    workspace = tmp_path / "workspace"
    private = workspace / "converter with spaces"
    program = private / "root/usr/lib/libreoffice/program"
    program.mkdir(parents=True)
    (private / "root/usr/lib/x86_64-linux-gnu").mkdir()
    for name in (
        "soffice", "soffice.bin", "oosplash", "fundamentalrc", "bootstraprc",
        "sofficerc", "unorc", "services.rdb", "types.rdb", "services/services.rdb",
        "libuno_cppuhelpergcc3.so.3", "libuno_cppu.so.3", "libuno_sal.so.3",
    ):
        path = program / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("fixture\n")
        if name in {"soffice", "soffice.bin", "oosplash"}:
            path.chmod(0o755)
    return workspace, private, program


def test_wrapper_prefers_program_and_preserves_existing_converter_interface(installation):
    workspace, private, program = installation
    program.joinpath("soffice").write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -c "
        + shlex.quote("import json, os, sys; print(json.dumps({'env': {k: os.environ[k] "
                        "for k in ('LD_LIBRARY_PATH', 'URE_BOOTSTRAP', 'XDG_CONFIG_HOME', "
                        "'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'TMPDIR')}, 'args': sys.argv[1:]}))")
        + ' "$@"\n'
    )
    import json
    result = load_tool().configure(private, workspace=workspace)
    wrapper = Path(result["wrapper"])
    output = json.loads(subprocess.check_output(
        [str(wrapper), "-env:UserInstallation=file:///workspace/own-profile", "--version"],
        text=True, env={"PATH": os.defpath, "LD_LIBRARY_PATH": "/vendor/ppu"},
    ))
    assert output["env"]["LD_LIBRARY_PATH"].split(":") == [
        str(program), str(private / "root/usr/lib/x86_64-linux-gnu")
    ]
    assert output["env"]["URE_BOOTSTRAP"] == "vnd.sun.star.pathname:" + str(program / "fundamentalrc")
    assert output["args"] == ["-env:UserInstallation=file:///workspace/own-profile", "--version"]
    for name in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "TMPDIR"):
        assert Path(output["env"][name]).is_dir()
        assert Path(output["env"][name]).is_relative_to(private)
    assert len({output["env"][name] for name in (
        "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "TMPDIR")}) == 4
    assert Path(result["receipt"]).is_file()
    assert result["previous_wrapper"] is None


def test_existing_wrapper_has_exact_executable_rollback_copy_and_idempotence(installation):
    workspace, private, _ = installation
    wrapper = private / "bin/soffice"
    wrapper.parent.mkdir()
    old = b"#!/bin/sh\nprintf 'previous version\\n'\n"
    wrapper.write_bytes(old)
    wrapper.chmod(0o750)
    tool = load_tool()
    first = tool.configure(private, workspace=workspace)
    previous = Path(first["previous_wrapper"])
    assert previous.read_bytes() == old
    assert previous.stat().st_mode & 0o777 == 0o750
    assert subprocess.check_output([str(previous)], text=True) == "previous version\n"
    second = tool.configure(private, workspace=workspace)
    assert second["changed"] is False
    assert second["wrapper_sha256"] == first["wrapper_sha256"]
    assert len(list(wrapper.parent.glob("soffice.previous-*"))) == 1


@pytest.mark.parametrize("relative", [
    "root/usr/lib/libreoffice/program/soffice.bin",
    "root/usr/lib/libreoffice/program/unorc",
    "root/usr/lib/libreoffice/program/libuno_cppuhelpergcc3.so.3",
    "root/usr/lib/libreoffice/program/services/services.rdb",
])
def test_missing_critical_file_leaves_wrapper_unmodified(installation, relative):
    workspace, private, _ = installation
    (private / relative).unlink()
    with pytest.raises(ValueError, match="required"):
        load_tool().configure(private, workspace=workspace)
    assert not (private / "bin").exists()


@pytest.mark.parametrize("relative", ["bin", "runtime", "runtime/config", "bin/soffice", "bin/soffice-configuration.json"])
def test_output_symlink_is_rejected_without_external_mutation(installation, tmp_path, relative):
    workspace, private, _ = installation
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep"
    sentinel.write_bytes(b"unchanged")
    target = private / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink|outside"):
        load_tool().configure(private, workspace=workspace)
    assert sentinel.read_bytes() == b"unchanged"
    assert list(outside.iterdir()) == [sentinel]


def test_critical_library_cannot_resolve_outside_package(installation, tmp_path):
    workspace, private, program = installation
    outside = tmp_path / "lib.so"
    outside.write_text("not the packaged library")
    library = program / "libuno_cppuhelpergcc3.so.3"
    library.unlink()
    library.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        load_tool().configure(private, workspace=workspace)


def test_relative_library_symlink_within_package_is_supported(installation):
    workspace, private, program = installation
    library = program / "libuno_cppuhelpergcc3.so.3"
    library.rename(program / "libuno_cppuhelpergcc3.so.3.0")
    library.symlink_to("libuno_cppuhelpergcc3.so.3.0")
    assert load_tool().configure(private, workspace=workspace)["changed"]


def test_executable_permission_required(installation):
    workspace, private, program = installation
    (program / "soffice.bin").chmod(0o644)
    with pytest.raises(ValueError, match="executable"):
        load_tool().configure(private, workspace=workspace)


def test_root_and_outside_workspace_rejected(installation, tmp_path):
    workspace, _, _ = installation
    for root in (workspace, tmp_path):
        with pytest.raises(ValueError, match="workspace"):
            load_tool().configure(root, workspace=workspace)


def test_colon_in_path_rejected_for_library_search_syntax(installation):
    workspace, private, _ = installation
    renamed = private.with_name("converter:wrong")
    private.rename(renamed)
    with pytest.raises(ValueError, match="colon"):
        load_tool().configure(renamed, workspace=workspace)


def test_cli_cannot_choose_a_different_workspace(installation):
    workspace, private, _ = installation
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--private-root", str(private), "--workspace", str(workspace)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not (private / "bin").exists()


def test_failed_atomic_replacement_preserves_original_and_rollback(installation, monkeypatch):
    workspace, private, _ = installation
    wrapper = private / "bin/soffice"
    wrapper.parent.mkdir()
    wrapper.write_bytes(b"original wrapper")
    original_replace = os.replace

    def fail_wrapper(source, target):
        if target == wrapper:
            raise OSError("simulated interrupted publication")
        return original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_wrapper)
    with pytest.raises(OSError, match="interrupted"):
        load_tool().configure(private, workspace=workspace)
    assert wrapper.read_bytes() == b"original wrapper"
    assert next(wrapper.parent.glob("soffice.previous-*")).read_bytes() == b"original wrapper"
    assert not list(wrapper.parent.glob(".*.pending-*"))


def test_special_output_file_is_rejected_without_blocking(installation):
    workspace, private, _ = installation
    (private / "bin").mkdir()
    os.mkfifo(private / "bin/.soffice-configuration.lock")
    with pytest.raises(ValueError, match="path type"):
        load_tool().configure(private, workspace=workspace)


def test_non_executable_identical_wrapper_is_repaired(installation):
    workspace, private, _ = installation
    tool = load_tool()
    first = tool.configure(private, workspace=workspace)
    wrapper = Path(first["wrapper"])
    wrapper.chmod(0o644)
    assert tool.configure(private, workspace=workspace)["changed"]
    assert os.access(wrapper, os.X_OK)
