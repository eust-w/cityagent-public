"""Offline explicit vector assets; no model, OCR, training, or network calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import zipfile

from bridge_evidence.contest.document_image_representation import (
    VERSION, MAX_BYTES, ImageRepresentationError, render_vector_png,
)


def prepare(document, directory, *, executable, members=None, dpi=200,
            timeout_seconds=30, max_pixels=16_000_000):
    document, directory = Path(document).resolve(strict=True), Path(directory)
    if directory.exists():
        raise ValueError("Choose a new output directory; prior assets must be preserved")
    directory.mkdir(parents=True)
    (directory / "assets").mkdir()
    original_sha = hashlib.sha256(document.read_bytes()).hexdigest()
    entries, results = [], []
    with zipfile.ZipFile(document) as archive:
        names = archive.namelist()
        selected = list(members) if members is not None else [name for name in names
            if name.startswith("word/media/") and PurePosixPath(name).suffix.lower() in {".emf", ".wmf"}]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Select at least one distinct original EMF/WMF member")
        for member in selected:
            row = {"zip_member": member, "status": "failed_unverified"}
            try:
                if (not member.startswith("word/media/") or "\\" in member
                        or posixpath.normpath(member) != member or names.count(member) != 1
                        or ".." in PurePosixPath(member).parts):
                    raise ImageRepresentationError("IMAGE_SOURCE_MEMBER_INVALID")
                info = archive.getinfo(member)
                if info.file_size > MAX_BYTES:
                    raise ImageRepresentationError("IMAGE_SOURCE_BYTES_EXCEEDED")
                blob = archive.read(info)
                row["original_blob_sha256"] = hashlib.sha256(blob).hexdigest()
                encoded, receipt = render_vector_png(blob, executable=executable, dpi=dpi,
                    max_pixels=max_pixels, timeout_seconds=timeout_seconds)
                relative = "assets/" + receipt["payload_image_sha256"] + ".png"
                target = directory / relative
                if target.exists() and target.read_bytes() != encoded:
                    raise ImageRepresentationError("IMAGE_DERIVED_PATH_COLLISION")
                target.write_bytes(encoded)
                entries.append({"zip_member": member, "png_path": relative, "conversion": receipt})
                row.update(status="converted_not_semantically_verified",
                           payload_image_sha256=receipt["payload_image_sha256"],
                           output_size=receipt["output_size"], png_path=relative)
            except ImageRepresentationError as exc:
                row["error_code"] = exc.code
                if exc.failure_receipt is not None:
                    row["failed_conversion_receipt"] = exc.failure_receipt
                if exc.failed_png is not None:
                    failure_directory = directory / "failures"
                    failure_directory.mkdir(exist_ok=True)
                    failed_relative = "failures/" + hashlib.sha256(exc.failed_png).hexdigest() + ".png"
                    (directory / failed_relative).write_bytes(exc.failed_png)
                    row["failed_png_path"] = failed_relative
            results.append(row)
    drifted = hashlib.sha256(document.read_bytes()).hexdigest() != original_sha
    complete = not drifted and all(row["status"] == "converted_not_semantically_verified" for row in results)
    manifest = {"version": VERSION, "document_sha256": original_sha, "entries": entries}
    report = {"version": VERSION, "source_document": str(document), "document_sha256": original_sha,
        "selected_count": len(selected), "prepared_count": len(entries), "preparation_complete": complete,
        "source_changed_during_preparation": drifted, "semantic_quality_passed": False,
        "ppu_runtime_verified": False, "results": results,
        "limitations": ["Successful raster conversion is not evidence of faithful semantics or model image understanding.",
                        "Unselected/unconverted image occurrences remain required by the existing workflow gate."]}
    for name, value in [("preparation-report.json", report), ("manifest.json", manifest)]:
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if drifted:
        raise ValueError("The original DOCX changed during preparation; assets cannot be accepted")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soffice", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--member", action="append", dest="members")
    selection.add_argument("--all-vector", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args()
    report = prepare(args.document, args.output, executable=args.soffice,
                     members=args.members, dpi=args.dpi, timeout_seconds=args.timeout_seconds)
    print(json.dumps({key: report[key] for key in ("selected_count", "prepared_count", "preparation_complete",
                                                  "semantic_quality_passed", "ppu_runtime_verified")}))
    return 0 if report["preparation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
