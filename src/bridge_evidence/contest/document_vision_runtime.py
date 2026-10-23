"""Bind an explicitly selected own-model image path to a report workflow.

Construction/verification read configuration only. The native CPU processor
and model are called exclusively through the report's admission/gateway.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import urlsplit

from .run_state import file_digest, json_digest


VERSION = "own-model-vision-runtime-v5"
PORTABLE_BINDING_VERSION = "catalog-and-native-process-model-binding-v2"


def _bound_endpoint(base_url, path_audit):
    """Bind generation to the explicitly audited local serving port."""
    try:
        if not isinstance(base_url, str) or base_url != base_url.strip() or any(c in base_url for c in "?#"):
            raise ValueError
        url = urlsplit(base_url)
        host, port = url.hostname, url.port
        if (url.scheme != "http" or url.username is not None or url.password is not None
                or url.path not in {"", "/"} or url.query or url.fragment
                or host not in {"127.0.0.1", "localhost", "::1"} or port is None):
            raise ValueError
        audit = json.loads(Path(path_audit).read_text(), object_pairs_hook=_unique)
        argv = audit["service_process"]["cmdline"]
        if (not isinstance(argv, list) or any(not isinstance(x, str) for x in argv)
                or argv.count("--host") != 1 or argv.count("--port") != 1):
            raise ValueError
        service_host = argv[argv.index("--host") + 1]
        service_port = int(argv[argv.index("--port") + 1])
        allowed = {"127.0.0.1": {"127.0.0.1", "localhost"}, "localhost": {"127.0.0.1", "localhost"},
                   "0.0.0.0": {"127.0.0.1", "localhost"}, "::1": {"::1", "localhost"},
                   "::": {"::1", "localhost"}}
        if host not in allowed.get(service_host, set()) or not 0 < service_port < 65536 or port != service_port:
            raise ValueError
        return f"http://{'[' + host + ']' if host == '::1' else host}:{port}"
    except (ValueError, TypeError, KeyError, IndexError, OSError):
        raise ValueError("Vision generation endpoint differs from the audited local service") from None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate vision configuration key")
        result[key] = value
    return result


class OwnModelVisionFactory:
    """A read-only preflight identity and a report-local backend factory."""

    def __init__(self, configuration: Path, *, model: str, model_identity: dict, base_url: str):
        from .document_multimodal_admission import NativeMultimodalAdmission
        from .document_vision_backend import MODEL, MAX_MODEL_LEN
        from .document_image_representation import PreparedVectorPngStore

        if model != MODEL or not isinstance(model_identity, dict) or not model_identity:
            raise ValueError("Vision must use the bound own-model deployment")
        self.configuration = Path(configuration).resolve(strict=True)
        self.configuration_sha256 = file_digest(self.configuration)
        value = json.loads(self.configuration.read_text(), object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite vision configuration")))
        required = {"processor_python", "path_audit", "runtime_environment"}
        if (not isinstance(value, dict) or not required <= set(value)
                or not set(value) <= required | {"image_representation_manifests", "model_binding", "modelinfo_cache"}
                or any(not isinstance(value[key], str) or not Path(value[key]).is_absolute()
                       for key in ("processor_python", "path_audit"))
                or not isinstance(value["runtime_environment"], dict)
                or set(value["runtime_environment"]) != {"PATH", "LD_LIBRARY_PATH"}
                or any(not isinstance(item, str) or (not item and (key == "PATH" or "model_binding" not in value))
                       for key, item in value["runtime_environment"].items())):
            raise ValueError("Vision configuration needs explicit native Python, source audit and PPU runtime paths")
        paths = value.get("image_representation_manifests", [])
        if (not isinstance(paths, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths)
                or len(paths) != len(set(paths))):
            raise ValueError("Image representation manifests must be distinct explicit absolute paths")
        self._vector_stores = {}
        for path in paths:
            store = PreparedVectorPngStore(Path(path))
            if store.document_sha256 in self._vector_stores:
                raise ValueError("More than one representation manifest binds the same original DOCX")
            self._vector_stores[store.document_sha256] = store
        self._admission_type = NativeMultimodalAdmission
        self._options = {"processor_python": Path(value["processor_python"]),
                         "path_audit": Path(value["path_audit"]),
                         "runtime_environment": copy.deepcopy(value["runtime_environment"])}
        if "model_binding" in value:
            # This explicit configuration only asserts the already checked
            # CLI catalog identity. It cannot supply or override deployment
            # identity, adapter paths or hashes. Native admission independently
            # checks local bytes and the audited/actual service command line.
            binding = value["model_binding"]
            deployment, adapter = model_identity.get("deployment"), model_identity.get("adapter")
            if (not isinstance(binding, dict) or set(binding) != {"base_model_root", "base_model_id"}
                    or any(not isinstance(binding.get(key), str) or not binding[key]
                           for key in ("base_model_root", "base_model_id"))
                    or not isinstance(deployment, dict) or not isinstance(adapter, dict)
                    or deployment.get("kind") != "dynamic_lora"
                    or deployment.get("binding_source") != "model_catalog.id+root+parent"
                    or any(binding.get(key) != deployment.get(key)
                           for key in ("base_model_root", "base_model_id"))):
                raise ValueError("Vision model binding differs from the CLI-verified model catalog identity")
            self._options["model_binding"] = {
                "schema_version": 1, "catalog_binding_source": deployment["binding_source"],
                "base_model_root": binding["base_model_root"], "base_model_id": binding["base_model_id"],
                "model": model, "adapter_path": deployment.get("adapter_path"),
                "adapter_weights_sha256": adapter.get("weights_sha256"),
                "adapter_config_sha256": adapter.get("config_sha256")}
        if "modelinfo_cache" in value:
            cache = value["modelinfo_cache"]
            if ("model_binding" not in self._options or not isinstance(cache, dict)
                    or set(cache) != {"path", "sha256"} or not isinstance(cache["path"], str)
                    or not Path(cache["path"]).is_absolute()):
                raise ValueError("Modelinfo cache requires explicit portable binding and a pinned absolute input")
            self._options["modelinfo_cache"] = copy.deepcopy(cache)
        self._probe = self._admission_type(self.configuration.parent / ".unused-vision-preflight",
                                           **self._options)
        self._native_identity = copy.deepcopy(self._probe.identity)
        self.endpoint = _bound_endpoint(base_url, self._options["path_audit"])
        self.identity = {"backend": "own-model", "status": "configured_not_executed",
            "version": VERSION, "model": model, "max_model_len": MAX_MODEL_LEN,
            "output_tokens": 512, "configuration_sha256": self.configuration_sha256,
            "endpoint": self.endpoint,
            "native_admission": copy.deepcopy(self._native_identity),
            "model_identity_sha256": json_digest(model_identity),
            "external_vision_api": False, "real_generation_verified": False,
            "limitations": ["Configured identity is not evidence of image generation or semantic accuracy.",
                            "Text-based semantic review of observations does not independently inspect pixels."]}
        if "model_binding" in self._options:
            self.identity["portable_model_binding"] = {
                "version": PORTABLE_BINDING_VERSION,
                "binding": copy.deepcopy(self._options["model_binding"])}
        if "modelinfo_cache" in self._options:
            self.identity["modelinfo_cache"] = copy.deepcopy(self._native_identity["modelinfo_cache"])
        if paths:
            self.identity["image_representations"] = [copy.deepcopy(store.identity)
                for _, store in sorted(self._vector_stores.items())]
        self.verify()

    def verify(self):
        if file_digest(self.configuration) != self.configuration_sha256:
            raise ValueError("Vision configuration changed during the bound run")
        self._probe.verify()
        if self._probe.identity != self._native_identity:
            raise ValueError("Native processor identity changed during the bound run")
        self.verify_endpoint(self.endpoint)
        for store in self._vector_stores.values():
            store.verify()

    def verify_endpoint(self, base_url):
        if _bound_endpoint(base_url, self._options["path_audit"]) != self.endpoint:
            raise ValueError("Vision generation endpoint changed during the bound run")

    def __call__(self, *, generate_observation, deadline, directory, manifest):
        from .document_vision_backend import OwnModelVisionBackend

        self.verify()
        admission = self._admission_type(Path(directory) / "admission", **self._options)
        if admission.identity != self._native_identity:
            raise ValueError("Vision worker identity differs from read-only preflight")
        # Native admission retains the complete portable binding/policy. Its
        # processor SHA binds those fields; the image backend accepts only its
        # strict five-field identity schema, not the full native schema.
        identity = {key: copy.deepcopy(self._native_identity[key]) for key in
                    ("model", "max_model_len", "processor_identity_sha256", "service_identity_sha256")}
        identity["model_identity_sha256"] = self.identity["model_identity_sha256"]
        backend = OwnModelVisionBackend(generate_observation, identity=identity,
            document_sha256=manifest["document_sha256"], deadline=deadline, output_tokens=512,
            vector_representations=self._vector_stores.get(manifest["document_sha256"]))
        return backend, admission
