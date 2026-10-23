"""Public defaults are unconfigured; no private deployment or model is accessed."""
from pathlib import Path
import hashlib

import pytest

from bridge_evidence.contest import document_native_processor as helper
from bridge_evidence.contest import document_multimodal_admission as admission


PUBLIC_PLACEHOLDER_ROOT = Path('/__cityagent_unconfigured_runtime__')


def test_public_defaults_have_no_bound_historical_deployment():
    assert helper.MODELINFO_ORIGIN['service_pid'] == 0
    assert helper.MODELINFO_ORIGIN['service_cmdline_sha256'] == '0' * 64
    paths = [helper.MODELINFO_ORIGIN['path'], helper.MODELINFO_ORIGIN['base_model_root'],
             helper.MODELINFO_SOURCE_ROOT, helper.CPU_METADATA_PYTHON,
             helper.CPU_METADATA_ALIAS_DIRECTORY, helper.CPU_METADATA_PLATFORM,
             helper.CPU_METADATA_CPUINFO]
    assert all(Path(path).is_relative_to(PUBLIC_PLACEHOLDER_ROOT) for path in paths)
    # Dependency byte pins remain active; only deployment provenance is absent.
    for pins in (helper.MODELINFO_SOURCE_PINS, helper.CPU_METADATA_FILE_PINS,
                 helper.CV2_IMPORT_SOURCE_PINS):
        assert pins
        assert all(len(value) == 64 and value != '0' * 64 for value in pins.values())


def test_public_historical_cache_is_rejected_before_any_file_read(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail('Unconfigured historical cache must not read input or dependency files')

    monkeypatch.setattr(helper, 'file_digest', forbidden)
    monkeypatch.setattr(helper, 'load_json', forbidden)
    with pytest.raises(ValueError, match='unconfigured in public source'):
        helper.modelinfo_cache_binding(tmp_path / 'absent.json', helper.MODELINFO_CACHE_SHA256,
                                       {}, helper.MODELINFO_ORIGIN['base_model_root'])


def test_public_cpu_metadata_policy_still_rejects_unmatched_runtime(monkeypatch):
    monkeypatch.setenv('PATH', '')
    with pytest.raises(ValueError, match='environment or interpreter'):
        helper.verify_cpu_metadata_policy()


def test_public_admission_pins_the_actual_helper_source():
    assert hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest() == admission.HELPER_SHA256
