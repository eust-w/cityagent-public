"""Pinned, run-local MinerU inputs, independent of the canonical evidence ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .run_state import atomic_json


def load_material_index(manifest, materials, source_path):
    from .document_material_index import DocumentMaterialIndex
    return DocumentMaterialIndex(manifest, materials, source_path=source_path,
                                 expected_document_sha256=manifest["document_sha256"])


def prepare_material_run(manifest, materials, source_path, directory):
    """Copy verified inputs once. Never modify or depend on the sidecar at runtime."""
    directory = Path(directory)
    raw = Path(source_path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["document_sha256"] or digest != materials.get("docx_sha256"):
        raise ValueError("MinerU source DOCX differs from the canonical document")
    destination = directory / "materials"
    destination.mkdir(parents=True, exist_ok=False)
    source = destination / "source.docx"
    source.write_bytes(raw)
    material_file = destination / "materials.json"
    atomic_json(material_file, materials)
    index = load_material_index(manifest, materials, source)
    index.save(destination / "reference-index.json")
    request = {"materials_path": str(material_file.resolve()),
               "materials_file_sha256": hashlib.sha256(material_file.read_bytes()).hexdigest(),
               "materials_source_docx": str(source.resolve()),
               "materials_source_sha256": digest}
    atomic_json(destination / "inputs.json", request)
    return index, request


def load_worker_material_index(manifest, request):
    if not request.get("materials_path"):
        if any(key.startswith("materials_") for key in request):
            raise ValueError("Incomplete MinerU input binding")
        return None
    raw = Path(request["materials_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != request.get("materials_file_sha256"):
        raise ValueError("MinerU materials changed after dispatch")
    if request.get("materials_source_sha256") != manifest["document_sha256"]:
        raise ValueError("MinerU source belongs to another document")
    return load_material_index(manifest, json.loads(raw), request["materials_source_docx"])
