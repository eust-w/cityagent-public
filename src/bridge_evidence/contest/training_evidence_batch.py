"""Explicit, hash-bound official-training text diagnostics; no dataset discovery.

The manifest declares the provenance scope, not proof of dataset membership.
Its input-only JSON entries reuse the established training-text parser. Text
diagnostics do not establish coverage of the original Word layout or images.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .run_state import file_digest


SCOPE = "official_training_text_only"
LIMITATIONS = [
    "Explicit official-training text diagnostic; provenance is declared by the supplied manifest",
    "Pre-extracted text does not establish original Word layout or image coverage",
]


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON keys are forbidden in a training batch manifest")
        value[key] = item
    return value


def _path_without_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Training batch paths must not contain symlinks")
    return path.resolve(strict=True)


def _outside(path, roots):
    if any(path == root or root in path.parents or path in root.parents for root in roots):
        raise ValueError("Training batch inputs and manifest must not overlap work/result paths")


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def load_training_evidence_batch(path, *, forbidden_roots=()):
    """Return ordered (ID, parsed evidence, original JSON) rows and an identity.

    No output/label fields are accepted, and no files are found automatically.
    ``forbidden_roots`` normally contains work_dir and result_dir.
    """
    from .divide_cli import _training_manifest

    path = _path_without_symlinks(path)
    roots = [Path(root).resolve() for root in forbidden_roots]
    _outside(path, roots)
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes, object_pairs_hook=_unique_object)
    if (type(raw) is not dict or set(raw) != {"schema_version", "scope", "entries"}
            or type(raw["schema_version"]) is not int or raw["schema_version"] != 1
            or raw["scope"] != SCOPE or type(raw["entries"]) is not list or not raw["entries"]):
        raise ValueError("Training batch requires schema_version=1, official_training_text_only and nonempty entries only")
    prepared, entries, paths, names = [], [], set(), set()
    for entry in raw["entries"]:
        if type(entry) is not dict or set(entry) != {"path", "sha256"} or not _sha(entry["sha256"]):
            raise ValueError("Training batch entries contain only path and lowercase SHA256")
        relative = entry["path"]
        if (type(relative) is not str or not relative or "\\" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
                or PurePosixPath(relative).as_posix() != relative or relative == "."):
            raise ValueError("Training batch entry path must be canonical, relative and stay inside its manifest directory")
        original = _path_without_symlinks(path.parent / relative)
        if not original.is_relative_to(path.parent) or not original.is_file():
            raise ValueError("Training batch entry must be a file inside its manifest directory")
        _outside(original, roots)
        if str(original).casefold() in paths:
            raise ValueError("Duplicate training batch source path")
        paths.add(str(original).casefold())
        expected = entry["sha256"]
        if file_digest(original) != expected:
            raise ValueError("Training batch source SHA256 mismatch")
        name, manifest = _training_manifest(original)
        if "\\" in name or name.casefold() in names:
            raise ValueError("Duplicate or unsafe training batch output ID")
        names.add(name.casefold())
        if manifest["source_json_sha256"] != expected or file_digest(original) != expected:
            raise ValueError("Training batch source changed while parsing")
        prepared.append((name, manifest, original))
        entries.append({"path": relative, "source_path": str(original), "sha256": expected, "evidence_id": name})
    identity = {"schema_version": 1, "scope": SCOPE, "manifest_path": str(path),
                "manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "entries": entries, "limitations": list(LIMITATIONS)}
    verify_training_evidence_batch(identity)
    return prepared, identity


def verify_training_evidence_batch(identity, *, original=None):
    """Check the bound manifest and sources; a worker checks only its own JSON."""
    manifest_path = _path_without_symlinks(identity["manifest_path"])
    if str(manifest_path) != identity["manifest_path"] or file_digest(manifest_path) != identity["manifest_sha256"]:
        raise ValueError("Training batch manifest changed during the bound run")
    entries = identity["entries"]
    if original is not None:
        original = _path_without_symlinks(original)
        entries = [row for row in entries if row["source_path"] == str(original)]
        if len(entries) != 1:
            raise ValueError("Worker original does not uniquely belong to the bound training batch")
    for row in entries:
        source = _path_without_symlinks(row["source_path"])
        if str(source) != row["source_path"] or file_digest(source) != row["sha256"]:
            raise ValueError("Training batch source changed during the bound run")
