"""Version-pinned, process-scoped MinerU native DOCX repair.

Keep Mammoth's OOXML-derived spans. Never rewrite legal spans to match a modal
row width. No installed-package edits; restore the hook even after failure.
Use in a dedicated worker process, never concurrently in a shared interpreter.
"""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
from pathlib import Path

SOURCE_HASH = 'fed8ba0dabc487162ecd9a509f0c7ea0387c61b9cf9fb7d07aa961bf852a16d0'

@contextmanager
def preserve_native_spans():
    from docvortex.analyzers.native.office.docx.tables import _DocxTables
    import inspect
    if (version('mineru'), version('docvortex')) != ('4.0.12', '0.5.16'):
        raise RuntimeError('Unvalidated parser version; repair requires MinerU 4.0.12 / DocVortex 0.5.16')
    path = Path(inspect.getfile(_DocxTables))
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASH:
        raise RuntimeError('Installed table implementation differs from reviewed source')
    original = _DocxTables._normalize_table_colspans
    if getattr(original, '_cityagent_repair', False):
        raise RuntimeError('Nested parser repair context is not supported')
    receipt = {'strategy': 'preserve_original_OOXML_spans', 'tables_seen': 0,
               'package_sha256': SOURCE_HASH, 'package_modified': False}
    def keep(self, markup):
        receipt['tables_seen'] += 1
        return markup
    keep._cityagent_repair = True
    _DocxTables._normalize_table_colspans = keep
    try:
        yield receipt
    finally:
        _DocxTables._normalize_table_colspans = original
