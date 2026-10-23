"""Explicit, content-bound vector raster assets; never image observations.

Preparation may use an explicitly selected local LibreOffice installation.
Serving only reads the resulting PNG assets: no converters, subprocesses,
network clients, OCR, model calls, or inferred replacement evidence.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import posixpath
import re
import struct
import subprocess
import tempfile
import time
import warnings
from xml.etree import ElementTree as ET

from PIL import Image


VERSION = "bound-vector-png-representation-v1"
MAX_BYTES = 32 * 1024 * 1024
PNG = b"\x89PNG\r\n\x1a\n"
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


class ImageRepresentationError(ValueError):
    def __init__(self, code, *, failure_receipt=None, failed_png=None):
        self.code = code
        self.failure_receipt = failure_receipt
        self.failed_png = failed_png
        super().__init__(code)


def _fail(code):
    raise ImageRepresentationError(code)


def vector_geometry(blob):
    """Read only declared vector bounds; do not guess missing dimensions."""
    if not isinstance(blob, bytes) or not 40 <= len(blob) <= MAX_BYTES:
        _fail("IMAGE_VECTOR_HEADER_INVALID")
    if (len(blob) >= 88 and blob[:4] == b"\x01\x00\x00\x00"
            and blob[40:44] == b" EMF"):
        header_size = struct.unpack_from("<I", blob, 4)[0]
        declared_bytes = struct.unpack_from("<I", blob, 48)[0]
        if not 88 <= header_size <= len(blob) or declared_bytes != len(blob):
            _fail("IMAGE_VECTOR_HEADER_INVALID")
        frame = list(struct.unpack_from("<4i", blob, 24))
        logical = list(struct.unpack_from("<4i", blob, 8))
        width_mm, height_mm = (frame[2] - frame[0]) / 100, (frame[3] - frame[1]) / 100
        result = {"content_type": "image/x-emf", "format": "EMF",
                  "declared_bounds": logical, "bounds_units": "EMF-device-units",
                  "physical_frame": frame, "physical_frame_units": "0.01mm"}
    elif blob[:4] == b"\xd7\xcd\xc6\x9a":
        if len(blob) < 40:
            _fail("IMAGE_VECTOR_HEADER_INVALID")
        words = struct.unpack_from("<11H", blob)
        checksum = 0
        for word in words[:10]:
            checksum ^= word
        _, _, left, top, right, bottom, inch, reserved, _ = struct.unpack_from("<IH4hHIH", blob)
        file_type, header_words, version, file_words = struct.unpack_from("<HHHI", blob, 22)
        if (checksum != words[10] or reserved != 0 or inch <= 0
                or file_type not in {1, 2} or header_words != 9 or version not in {0x100, 0x300}
                or file_words * 2 != len(blob) - 22):
            _fail("IMAGE_VECTOR_HEADER_INVALID")
        frame = [left, top, right, bottom]
        width_mm, height_mm = (right - left) * 25.4 / inch, (bottom - top) * 25.4 / inch
        result = {"content_type": "image/x-wmf", "format": "WMF",
                  "declared_bounds": frame, "bounds_units": f"1/{inch}-inch",
                  "physical_frame": frame, "physical_frame_units": f"1/{inch}-inch"}
    else:
        _fail("IMAGE_VECTOR_FORMAT_UNSUPPORTED")
    if (not all(math.isfinite(v) and 0 < v <= 10000 for v in (width_mm, height_mm))
            or result["declared_bounds"][2] <= result["declared_bounds"][0]
            or result["declared_bounds"][3] <= result["declared_bounds"][1]):
        _fail("IMAGE_VECTOR_DIMENSIONS_INVALID")
    result["canvas_mm"] = [width_mm, height_mm]
    return result


def _png_dimensions(blob, max_pixels):
    if not isinstance(blob, bytes) or not blob.startswith(PNG) or len(blob) > MAX_BYTES:
        _fail("IMAGE_REPRESENTATION_PNG_INVALID")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(blob)) as image:
                size = image.size
                if (image.format != "PNG" or getattr(image, "n_frames", 1) != 1
                        or size[0] * size[1] > max_pixels):
                    _fail("IMAGE_REPRESENTATION_PNG_INVALID")
                image.verify()
            with Image.open(io.BytesIO(blob)) as image:
                image.load()
                if image.mode not in {"L", "LA", "RGB", "RGBA", "P"} or image.getexif().get(274, 1) != 1:
                    _fail("IMAGE_REPRESENTATION_PNG_INVALID")
                rgba = image.convert("RGBA")
                visible = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("RGB")
                if all(low == high for low, high in visible.getextrema()):
                    _fail("IMAGE_REPRESENTATION_BLANK")
        return list(size)
    except ImageRepresentationError:
        raise
    except Exception:
        _fail("IMAGE_REPRESENTATION_PNG_INVALID")


def _bound_fodg(imported, blob, geometry):
    """Fit the actual imported Draw object, retaining the embedded original."""
    if len(imported) > 64 * 1024 * 1024 or b"<!DOCTYPE" in imported or b"<!ENTITY" in imported:
        _fail("IMAGE_CONVERTER_WRAPPER_INVALID")
    ns = {name: f"urn:oasis:names:tc:opendocument:xmlns:{suffix}:1.0"
          for name, suffix in {"office": "office", "draw": "drawing", "style": "style",
                               "svg": "svg-compatible", "fo": "xsl-fo-compatible"}.items()}
    try:
        # LibreOffice flat-ODF detection needs its normal namespace prefixes.
        for _, (prefix, uri) in ET.iterparse(io.BytesIO(imported), events=["start-ns"]):
            ET.register_namespace(prefix, uri)
        root = ET.fromstring(imported)
        pages = root.findall("./office:body/office:drawing/draw:page", ns)
        if len(pages) != 1 or len(pages[0]) != 1 or pages[0][0].tag != "{" + ns["draw"] + "}frame":
            _fail("IMAGE_CONVERTER_WRAPPER_INVALID")
        page, frame = pages[0], pages[0][0]
        originals = []
        for image in frame.findall("draw:image", ns):
            data = image.find("office:binary-data", ns)
            raw = base64.b64decode("".join((data.text or "").split()), validate=True) if data is not None else b""
            if raw == blob:
                originals.append(image)
            else:
                frame.remove(image)  # discard the imported fallback PNG, not the vector
        if len(originals) != 1 or len(frame) != 1:
            _fail("IMAGE_CONVERTER_WRAPPER_INVALID")
        master_name = page.get("{" + ns["draw"] + "}master-page-name")
        masters = [x for x in root.findall("./office:master-styles/style:master-page", ns)
                   if x.get("{" + ns["style"] + "}name") == master_name]
        if len(masters) != 1:
            _fail("IMAGE_CONVERTER_WRAPPER_INVALID")
        layout_name = masters[0].get("{" + ns["style"] + "}page-layout-name")
        layouts = [x for x in root.findall("./office:automatic-styles/style:page-layout", ns)
                   if x.get("{" + ns["style"] + "}name") == layout_name]
        if len(layouts) != 1:
            _fail("IMAGE_CONVERTER_WRAPPER_INVALID")
        properties = layouts[0].find("style:page-layout-properties", ns)
        if properties is None:
            _fail("IMAGE_CONVERTER_WRAPPER_INVALID")
        width, height = geometry["canvas_mm"]
        for key, value in {"page-width": f"{width}mm", "page-height": f"{height}mm",
                           "margin-left": "0mm", "margin-right": "0mm",
                           "margin-top": "0mm", "margin-bottom": "0mm"}.items():
            properties.set("{" + ns["fo"] + "}" + key, value)
        properties.set("{" + ns["style"] + "}print-orientation", "landscape" if width > height else "portrait")
        for key, value in {"x": "0mm", "y": "0mm", "width": f"{width}mm", "height": f"{height}mm"}.items():
            frame.set("{" + ns["svg"] + "}" + key, value)
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)
    except ImageRepresentationError:
        raise
    except (ValueError, TypeError, ET.ParseError, RecursionError):
        _fail("IMAGE_CONVERTER_WRAPPER_INVALID")


def render_vector_png(blob, *, executable, dpi=200, max_pixels=16_000_000,
                      timeout_seconds=30):
    """Offline-only conversion with declared size and a checkable receipt.

    A successful return proves decodability and the requested canvas dimensions,
    not visual fidelity. Font substitution/renderer differences need pixel review.
    """
    if (type(dpi) is not int or not 72 <= dpi <= 600 or type(max_pixels) is not int
            or max_pixels <= 0 or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        _fail("IMAGE_CONVERTER_CONFIGURATION_INVALID")
    geometry = vector_geometry(blob)
    size = [math.ceil(mm / 25.4 * dpi) for mm in geometry["canvas_mm"]]
    if size[0] * size[1] > max_pixels:
        _fail("IMAGE_REPRESENTATION_PIXELS_EXCEEDED")
    path = Path(executable)
    if not path.is_absolute() or not path.is_file():
        _fail("IMAGE_CONVERTER_UNAVAILABLE")
    resolved = path.resolve(strict=True)
    executable_sha = _sha(resolved.read_bytes())
    start = time.monotonic()
    try:
        version = subprocess.run([str(resolved), "--version"], capture_output=True,
                                 timeout=timeout_seconds, check=True)
    except (OSError, subprocess.SubprocessError):
        _fail("IMAGE_CONVERTER_UNAVAILABLE")
    version_text = version.stdout.decode("utf-8", "replace").strip()
    if not version_text.startswith("LibreOffice ") or len(version_text) > 1000:
        _fail("IMAGE_CONVERTER_IDENTITY_INVALID")
    with tempfile.TemporaryDirectory(prefix="cityagent-vector-") as temporary:
        root = Path(temporary)
        original = root / ("original." + geometry["format"].lower())
        original.write_bytes(blob)
        imported_directory = root / "imported"
        imported_directory.mkdir()
        remaining = timeout_seconds - (time.monotonic() - start)
        if remaining <= 0:
            _fail("IMAGE_CONVERTER_TIMEOUT")
        import_argv = [str(resolved), "-env:UserInstallation=" + (root / "profile-import").as_uri(),
                       "--headless", "--convert-to", "fodg", "--outdir", str(imported_directory), str(original)]
        try:
            imported_result = subprocess.run(import_argv, capture_output=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            _fail("IMAGE_CONVERTER_TIMEOUT")
        except OSError:
            _fail("IMAGE_CONVERTER_FAILED")
        imported_path = imported_directory / "original.fodg"
        if (imported_result.returncode != 0 or list(imported_directory.iterdir()) != [imported_path]
                or imported_path.is_symlink() or not imported_path.is_file()
                or imported_path.stat().st_size > 64 * 1024 * 1024):
            _fail("IMAGE_CONVERTER_FAILED")
        imported_bytes = imported_path.read_bytes()
        src = root / "bound.fodg"
        source = _bound_fodg(imported_bytes, blob, geometry)
        src.write_bytes(source)
        output = root / "output"
        output.mkdir()
        options = {"PixelWidth": {"type": "long", "value": size[0]},
                   "PixelHeight": {"type": "long", "value": size[1]}}
        filter_name = "png:draw_png_Export:" + json.dumps(options, separators=(",", ":"))
        argv = [str(resolved), "-env:UserInstallation=" + (root / "profile").as_uri(),
                "--headless", "--convert-to", filter_name, "--outdir", str(output), str(src)]
        remaining = timeout_seconds - (time.monotonic() - start)
        if remaining <= 0:
            _fail("IMAGE_CONVERTER_TIMEOUT")
        try:
            result = subprocess.run(argv, capture_output=True, timeout=remaining)
        except subprocess.TimeoutExpired:
            _fail("IMAGE_CONVERTER_TIMEOUT")
        except OSError:
            _fail("IMAGE_CONVERTER_FAILED")
        files = list(output.iterdir())
        target = output / "bound.png"
        if (result.returncode != 0 or files != [target] or target.is_symlink()
                or not target.is_file() or target.stat().st_size > MAX_BYTES):
            _fail("IMAGE_CONVERTER_FAILED")
        encoded = target.read_bytes()
    if _sha(resolved.read_bytes()) != executable_sha:
        _fail("IMAGE_CONVERTER_CHANGED")
    failure_receipt = {"version": VERSION, "status": "failed_unverified",
        "original_blob_sha256": _sha(blob), "original_content_type": geometry["content_type"],
        "converter_version": version_text, "converter_executable_sha256": executable_sha,
        "imported_fodg_sha256": _sha(imported_bytes), "wrapper_sha256": _sha(source),
        "requested_size": size, "dpi": dpi,
        "output_sha256": _sha(encoded), "output_bytes": len(encoded),
        "stdout_sha256": _sha(result.stdout), "stderr_sha256": _sha(result.stderr),
        "model_generation_attempted": False, "semantic_quality_passed": False}
    try:
        if _png_dimensions(encoded, max_pixels) != size:
            _fail("IMAGE_CONVERTER_DIMENSIONS_MISMATCH")
    except ImageRepresentationError as exc:
        failure_receipt["error_code"] = exc.code
        failure_receipt["receipt_sha256"] = _sha(_packed(failure_receipt))
        raise ImageRepresentationError(exc.code, failure_receipt=failure_receipt, failed_png=encoded) from None
    receipt = {"version": VERSION, "status": "converted_not_semantically_verified",
        "original_blob_sha256": _sha(blob), "original_bytes": len(blob),
        "original_content_type": geometry["content_type"], "geometry": geometry,
        "converter": {"name": "LibreOffice", "version": version_text,
                      "executable": str(resolved), "executable_sha256": executable_sha,
                      "identity_scope": "selected-launcher-bytes-and-reported-version; not all renderer libraries or fonts"},
        "method": "embedded-original-metafile-in-exact-canvas-fodg/draw_png_Export",
        "imported_fodg_sha256": _sha(imported_bytes), "wrapper_sha256": _sha(source), "dpi": dpi,
        "coordinate_mapping": {"source_space": "declared-vector-physical-canvas",
            "source_canvas_mm": geometry["canvas_mm"], "target_space": "derived-raster-pixels-top-left",
            "target_size": size, "origin": [0, 0], "crop": None,
            "pixels_per_mm": [size[i] / geometry["canvas_mm"][i] for i in (0, 1)],
            "physical_measurements_verified": False},
        "output_content_type": "image/png", "output_size": size,
        "payload_image_sha256": _sha(encoded), "payload_image_bytes": len(encoded),
        "stdout_sha256": _sha(result.stdout), "stderr_sha256": _sha(result.stderr),
        "import_stdout_sha256": _sha(imported_result.stdout),
        "import_stderr_sha256": _sha(imported_result.stderr),
        "stderr_nonempty": bool(result.stderr.strip()),
        "limitations": ["Rendering may substitute fonts or interpret metafile records differently.",
                        "Conversion does not establish fidelity, visual semantics, or measured dimensions."]}
    receipt["receipt_sha256"] = _sha(_packed(receipt))
    return encoded, receipt


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            _fail("IMAGE_REPRESENTATION_MANIFEST_INVALID")
        value[key] = item
    return value


class PreparedVectorPngStore:
    """Read-only vector PNG assets bound to document, member and original bytes."""

    def __init__(self, manifest_path, *, max_pixels=16_000_000):
        self.path = Path(manifest_path)
        if (not self.path.is_absolute() or self.path.is_symlink() or not self.path.is_file()
                or self.path.stat().st_size > MAX_BYTES or type(max_pixels) is not int or max_pixels <= 0):
            _fail("IMAGE_REPRESENTATION_MANIFEST_INVALID")
        self.path = self.path.resolve(strict=True)
        self.max_pixels = max_pixels
        self.manifest_sha256 = _sha(self.path.read_bytes())
        try:
            value = json.loads(self.path.read_bytes(), object_pairs_hook=_unique,
                               parse_constant=lambda _: _fail("IMAGE_REPRESENTATION_MANIFEST_INVALID"))
        except (ValueError, UnicodeError, RecursionError):
            _fail("IMAGE_REPRESENTATION_MANIFEST_INVALID")
        if (not isinstance(value, dict) or set(value) != {"version", "document_sha256", "entries"}
                or value["version"] != VERSION or not isinstance(value["document_sha256"], str)
                or not _HASH.fullmatch(value["document_sha256"]) or not isinstance(value["entries"], list)):
            _fail("IMAGE_REPRESENTATION_MANIFEST_INVALID")
        self.document_sha256, self._entries = value["document_sha256"], {}
        for entry in value["entries"]:
            if (not isinstance(entry, dict) or set(entry) != {"zip_member", "png_path", "conversion"}
                    or not isinstance(entry["zip_member"], str) or entry["zip_member"] in self._entries
                    or not entry["zip_member"].startswith("word/")
                    or posixpath.normpath(entry["zip_member"]) != entry["zip_member"]
                    or "\\" in entry["zip_member"]):
                _fail("IMAGE_REPRESENTATION_MANIFEST_INVALID")
            self._entries[entry["zip_member"]] = deepcopy(entry)
            self._asset(entry)
        self.identity = {"version": VERSION, "manifest_sha256": self.manifest_sha256,
                         "document_sha256": self.document_sha256, "entries": len(self._entries),
                         "conversion_is_semantic_verification": False}

    def _asset(self, entry):
        relative = entry["png_path"]
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts
                or posixpath.normpath(relative) != relative):
            _fail("IMAGE_REPRESENTATION_PATH_INVALID")
        path = self.path.parent / relative
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES
                or not path.resolve(strict=True).is_relative_to(self.path.parent)):
            _fail("IMAGE_REPRESENTATION_PATH_INVALID")
        receipt = deepcopy(entry["conversion"])
        if not isinstance(receipt, dict):
            _fail("IMAGE_REPRESENTATION_RECEIPT_INVALID")
        receipt_sha = receipt.pop("receipt_sha256", None)
        if receipt_sha != _sha(_packed(receipt)):
            _fail("IMAGE_REPRESENTATION_RECEIPT_INVALID")
        receipt["receipt_sha256"] = receipt_sha
        if (receipt.get("version") != VERSION or receipt.get("status") != "converted_not_semantically_verified"
                or receipt.get("output_content_type") != "image/png"
                or receipt.get("original_content_type") not in {"image/x-emf", "image/x-wmf"}
                or not _HASH.fullmatch(str(receipt.get("original_blob_sha256", "")))
                or type(receipt.get("original_bytes")) is not int or not 0 < receipt["original_bytes"] <= MAX_BYTES
                or not isinstance(receipt.get("geometry"), dict)
                or not isinstance(receipt.get("converter"), dict)
                or not isinstance(receipt["converter"].get("version"), str)
                or not receipt["converter"]["version"]
                or not isinstance(receipt.get("method"), str) or not receipt["method"]
                or not isinstance(receipt.get("coordinate_mapping"), dict)
                or receipt["coordinate_mapping"].get("target_space") != "derived-raster-pixels-top-left"
                or receipt["coordinate_mapping"].get("target_size") != receipt.get("output_size")
                or receipt["coordinate_mapping"].get("physical_measurements_verified") is not False):
            _fail("IMAGE_REPRESENTATION_RECEIPT_INVALID")
        blob = path.read_bytes()
        if (receipt.get("payload_image_sha256") != _sha(blob)
                or receipt.get("payload_image_bytes") != len(blob)
                or receipt.get("output_size") != _png_dimensions(blob, self.max_pixels)):
            _fail("IMAGE_REPRESENTATION_CHANGED")
        return blob, receipt

    def verify(self):
        if (self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_size > MAX_BYTES
                or _sha(self.path.read_bytes()) != self.manifest_sha256):
            _fail("IMAGE_REPRESENTATION_CHANGED")
        for entry in self._entries.values():
            self._asset(entry)

    def resolve(self, blob, *, member, document_sha256):
        if (self.path.is_symlink() or not self.path.is_file() or self.path.stat().st_size > MAX_BYTES
                or _sha(self.path.read_bytes()) != self.manifest_sha256):
            _fail("IMAGE_REPRESENTATION_CHANGED")
        if document_sha256 != self.document_sha256 or member not in self._entries:
            _fail("IMAGE_REPRESENTATION_UNAVAILABLE")
        encoded, receipt = self._asset(self._entries[member])
        if (receipt["original_blob_sha256"] != _sha(blob)
                or receipt["original_bytes"] != len(blob)
                or receipt["geometry"] != vector_geometry(blob)):
            _fail("IMAGE_REPRESENTATION_SOURCE_MISMATCH")
        return encoded, {"store": deepcopy(self.identity), "conversion": receipt,
                         "asset_zip_member": member}
