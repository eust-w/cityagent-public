"""Explicit model preparation; fast reuse requires a kernel read-only mount.

An operator pins the preparation receipt digest. Mutable model files are always
rehash-verified, regardless of timestamps. No permissions or mounts are changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .run_state import MODEL_IDENTITY_SUFFIXES, atomic_json, file_digest, file_digests


def readonly_mount(path):
    """Check the actual resolved filesystem, not Unix file permission bits."""
    try:
        return bool(os.statvfs(Path(path).resolve()).f_flag & os.ST_RDONLY)
    except (OSError, AttributeError):
        return False


def _stamp(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {'path': str(path), 'device': stat.st_dev, 'inode': stat.st_ino,
            'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns}


def _inventory(manifest):
    return {
        label: {p.relative_to(folder).as_posix(): _stamp(p)
                for p in sorted(Path(folder).rglob('*'))
                if p.is_file() and p.suffix in MODEL_IDENTITY_SUFFIXES}
        for label, folder in (('base', manifest.base), ('adapter', manifest.adapter),
                              ('tokenizer', manifest.tokenizer), ('merged', manifest.merged))
        if folder
    }


def prepare_identity(manifest, output):
    """Hash actual bytes, then publish an operator-pinned preparation receipt."""
    before = _inventory(manifest)
    identity = manifest.identity()
    if before != _inventory(manifest):
        raise ValueError('model changed during identity preparation')
    atomic_json(output, {'schema_version': 1, 'identity': identity, 'inventory': before})
    return file_digest(Path(output))


def load_identity(manifest, receipt, *, expected_sha256):
    """Return the exact original identity, never grant a new model identity."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError('prepared identity requires an explicitly pinned SHA256')
    raw = Path(receipt).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('prepared identity receipt digest changed')
    value = json.loads(raw)
    if value.get('schema_version') != 1 or value['identity']['manifest'] != manifest.as_dict():
        raise ValueError('prepared model manifest changed')
    manifest.validate()
    current = _inventory(manifest)
    expected = value['inventory']
    if (set(current) != set(expected)
            or any(set(current[k]) != set(expected[k]) for k in current)):
        raise ValueError('model inventory changed')
    checked, mutable = {}, []
    for label, files in current.items():
        for name, stamp in files.items():
            if stamp != expected[label][name]:
                raise ValueError('model file identity changed')
            digest = value['identity']['files'][label][name]
            if stamp['path'] in checked:
                if checked[stamp['path']] != digest:
                    raise ValueError('model aliases contain conflicting digests')
                continue
            checked[stamp['path']] = digest
            if not readonly_mount(Path(stamp['path'])):
                mutable.append((Path(stamp['path']), digest))
    actual = file_digests([path for path, _ in mutable], file_digest)
    if any(found != digest for found, (_, digest) in zip(actual, mutable)):
        raise ValueError('mutable model content changed')
    if current != _inventory(manifest):
        raise ValueError('model changed while validating prepared identity')
    return value['identity']
