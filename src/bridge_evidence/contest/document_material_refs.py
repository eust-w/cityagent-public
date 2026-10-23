"""Map material addresses to immutable evidence ranges without changing claims."""
from copy import deepcopy
import re

from .run_state import json_digest


class MaterialRefNormalizationError(ValueError):
    def __init__(self, message, *, path, code="SOURCE_MATERIAL", example=None):
        super().__init__(message)
        self.path, self.code = path, code
        self.example = example or {"evidence_id": "Use an exact source ID returned by the reading tool"}


def material_index_identity(manifest, material_index):
    if material_index is None:
        return None
    identity = deepcopy(material_index.identity)
    if (identity.get("document_sha256") != manifest["document_sha256"]
            or identity.get("manifest_sha256") != json_digest(manifest)
            or not isinstance(identity.get("materials_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", identity["materials_sha256"]) is None):
        raise ValueError("Material index differs from the current immutable document/manifest")
    return identity


def normalize_material_refs(refs, manifest, material_index, *, path="source_refs"):
    """Return canonical refs plus host-owned alias provenance; never mark reads.

    An alias denotes exactly the index's returned spans. Offsets on alias text
    are ambiguous with its presentation and are refused. Canonical ranges keep
    their original checks in the receiving store.
    """
    identity = material_index_identity(manifest, material_index)
    if not isinstance(refs, list):
        return deepcopy(refs), []
    units = {u["id"]: u["text"] for u in manifest["units"]}
    canonical, provenance = [], []
    for index, ref in enumerate(refs):
        here = f"{path}[{index}]"
        if not isinstance(ref, dict) or "image_id" in ref or not isinstance(ref.get("evidence_id"), str):
            canonical.append(deepcopy(ref))
            continue
        identifier = ref["evidence_id"]
        if identifier in units:
            value = deepcopy(ref)
            if "start" not in value and "end" not in value:
                value.update(start=0, end=len(units[identifier]))
            canonical.append(value)
            continue
        if material_index is None:
            canonical.append(deepcopy(ref))
            continue
        if set(ref) != {"evidence_id"}:
            raise MaterialRefNormalizationError("Material aliases take only evidence_id; use an exact E range from a tool for a narrower quote.",
                path=here, code="SOURCE_ALIAS_RANGE", example={"evidence_id": identifier})
        try:
            resolved = material_index.resolve(identifier, include_context=False)
        except (ValueError, KeyError) as exc:
            raise MaterialRefNormalizationError(str(exc), path=here + ".evidence_id") from exc
        if (not isinstance(resolved, dict) or resolved.get("ok") is not True
                or any(resolved.get(k) != v for k, v in identity.items())
                or not isinstance(resolved.get("spans"), list) or not resolved["spans"]
                or resolved.get("unresolved")):
            raise MaterialRefNormalizationError("Material reference has no unambiguous canonical source mapping; reread the reported E source.",
                path=here + ".evidence_id")
        spans = []
        for span in resolved["spans"]:
            if not isinstance(span, dict):
                raise MaterialRefNormalizationError("Material span must be an exact source range.", path=here)
            evidence_id, start, end = span.get("evidence_id"), span.get("start"), span.get("end")
            if (evidence_id not in units or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(units[evidence_id])
                    or span.get("text") != units[evidence_id][start:end]):
                raise MaterialRefNormalizationError("Material span differs from the immutable canonical source.", path=here)
            item = {"evidence_id": evidence_id, "start": start, "end": end}
            if item not in spans:
                spans.append(item)
        for span in spans:
            if span not in canonical:
                canonical.append(span)
        provenance.append({"source_ref": {"evidence_id": material_index.qualify_ref(identifier)},
                           "canonical_refs": spans, "identity": deepcopy(identity)})
    return canonical, provenance


def retain_material_provenance(previous, added, canonical_refs):
    """Preserve origin on idempotent canonical resaves; discard obsolete refs."""
    retained = []
    for row in [*previous, *added]:
        if all(ref in canonical_refs for ref in row.get("canonical_refs", [])) and row not in retained:
            retained.append(deepcopy(row))
    return retained


def validate_material_provenance(rows, refs, manifest, material_index):
    if not isinstance(rows, list):
        raise ValueError("Saved material provenance must be an array")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"source_ref", "canonical_refs", "identity"}:
            raise ValueError("Saved material provenance has invalid fields")
        normalized, expected = normalize_material_refs([row["source_ref"]], manifest, material_index)
        if len(expected) != 1 or expected[0] != row or normalized != row["canonical_refs"]:
            raise ValueError("Saved material provenance differs from the pinned index")
        if not normalized or any(ref not in refs for ref in normalized):
            raise ValueError("Saved material provenance refers to another record's evidence")


def validate_entry_material_provenance(state, manifest, material_index):
    """Validate a child snapshot before importing any of its reading/claims."""
    identity = material_index_identity(manifest, material_index)
    if state.get("identity", {}).get("material_identity") != identity:
        raise ValueError("Child material identity differs from this document's pinned index")
    provenance = state.get("material_provenance", {})
    entries = state.get("entries", {})
    if isinstance(entries, list):
        entries = {entry["id"]: entry for entry in entries}
    if not isinstance(provenance, dict) or set(provenance) - set(entries):
        raise ValueError("Child material provenance has no corresponding entry")
    for entry_id, rows in provenance.items():
        validate_material_provenance(rows, entries[entry_id]["source_refs"], manifest, material_index)
    return deepcopy(provenance)
