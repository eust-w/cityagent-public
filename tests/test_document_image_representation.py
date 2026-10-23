"""Offline pixel/binding contracts; synthetic receipts never prove model quality."""
import base64
from copy import deepcopy
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
from unittest.mock import patch

from PIL import Image, ImageDraw
import pytest

from bridge_evidence.contest import document_multimodal_admission as native
from bridge_evidence.contest.document_image_representation import (
    VERSION, ImageRepresentationError, PreparedVectorPngStore,
    render_vector_png, vector_geometry, _bound_fodg,
)
from bridge_evidence.contest.document_vision_backend import OwnModelVisionBackend, VisionObservationError, json_digest
from bridge_evidence.contest.document_vision_runtime import OwnModelVisionFactory
from test_document_vision_backend import arguments, backend, FakeGeneration, identity
from test_document_vision_runtime import StubNativeAdmission, MODEL_IDENTITY
from test_document_tools import _image_document
from bridge_evidence.contest.document_tools import DocumentTools


ROOT = Path(__file__).resolve().parents[1]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def png(size=(20, 12)):
    image = Image.new("RGB", size, (123, 45, 67))
    ImageDraw.Draw(image).line((0, 0, size[0] - 1, size[1] - 1), fill="white", width=1)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def jpeg():
    """Rebuild a nonuniform image without any private training-report assets."""
    image = Image.new("RGB", (96, 80))
    image.putdata([((x * 7 + y * 3) % 256, (x * 2 + y * 11) % 256,
                    ((x // 8 + y // 8) % 2) * 255)
                   for y in range(image.height) for x in range(image.width)])
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=91, subsampling=0)
    return stream.getvalue()


def emf():
    blob = bytearray(88)
    struct.pack_into("<II4i4i", blob, 0, 1, 88, 0, 0, 20, 12, 0, 0, 2000, 1200)
    blob[40:44] = b" EMF"
    struct.pack_into("<I", blob, 48, len(blob))
    return bytes(blob)


def assets(tmp_path, *, source=None, document_sha="a" * 64, member="word/media/image1.emf"):
    source = source or emf()
    encoded = png()
    (tmp_path / "asset.png").write_bytes(encoded)
    geometry = vector_geometry(source)
    # Fabricated structural fixture, explicitly not a LibreOffice receipt.
    receipt = {"version": VERSION, "status": "converted_not_semantically_verified",
        "original_blob_sha256": sha(source), "original_bytes": len(source),
        "original_content_type": geometry["content_type"], "geometry": geometry,
        "converter": {"version": "synthetic-no-renderer-executed"},
        "method": "synthetic-contract-only", "coordinate_mapping": {
            "target_space": "derived-raster-pixels-top-left", "target_size": [20, 12],
            "physical_measurements_verified": False},
        "output_content_type": "image/png", "output_size": [20, 12],
        "payload_image_sha256": sha(encoded), "payload_image_bytes": len(encoded)}
    receipt["receipt_sha256"] = json_digest(receipt)
    value = {"version": VERSION, "document_sha256": document_sha,
        "entries": [{"zip_member": member, "png_path": "asset.png", "conversion": receipt}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    return source, encoded, path, value


def test_generated_jpeg_decodes_to_exact_pixels_without_model_call():
    blob = jpeg()
    fake = FakeGeneration()
    source, options = arguments(blob)
    options["image"]["target"] = "media/image70.jpeg"
    payload, provenance = backend(fake).prepare_request(source, **options)
    sent = base64.b64decode(payload["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(blob)) as original, Image.open(io.BytesIO(sent)) as result:
        assert original.format == "JPEG"
        assert result.format == "PNG" and result.size == original.size == (96, 80)
        assert result.tobytes() == original.convert("RGB").tobytes()
    rep = provenance["representation"]
    assert rep["original_content_type"] == "image/jpeg" and rep["mime"] == "image/png"
    assert rep["original_blob_sha256"] == sha(blob) and rep["payload_image_sha256"] == sha(sent)
    assert rep["raw_bytes_unchanged"] is False and rep["resize"] is None
    assert provenance["source"]["zip_member"] == "word/media/image70.jpeg"
    assert not fake.calls


def test_jpeg_crop_uses_original_decoded_pixels_and_preserves_occurrence():
    blob = jpeg()
    _, options = arguments(blob)
    options["region"] = {"x": 13, "y": 17, "width": 41, "height": 53}
    options["image"]["target"] = "media/image70.jpeg"
    payload, provenance = backend().prepare_request(blob, **options)
    sent = base64.b64decode(payload["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(blob)) as original, Image.open(io.BytesIO(sent)) as result:
        assert result.tobytes() == original.crop((13, 17, 54, 70)).convert("RGBA").tobytes()
    assert provenance["representation"]["coordinate_space"] == "original-decoded-pixels-top-left"
    assert provenance["source"]["image_id"] == options["image"]["id"]


@pytest.mark.parametrize("kind", ["orientation", "pixels", "broken", "request-bytes"])
def test_jpeg_invalid_or_over_budget_never_generates(kind):
    image = Image.new("RGB", (8, 6))
    stream = io.BytesIO()
    exif = Image.Exif()
    if kind == "orientation":
        exif[274] = 6
    image.save(stream, format="JPEG", exif=exif)
    blob = stream.getvalue()
    if kind == "broken":
        blob = blob[:31]
    fake = FakeGeneration()
    _, options = arguments(blob)
    options["image"]["target"] = "media/test.jpeg"
    limits = {"max_pixels": 47} if kind == "pixels" else {"max_request_bytes": 1} if kind == "request-bytes" else {}
    with pytest.raises(VisionObservationError):
        backend(fake, **limits)(blob, **options)
    assert not fake.calls


def test_bound_vector_png_has_separate_source_and_derived_hashes_and_coordinate_space(tmp_path):
    original, encoded, path, _ = assets(tmp_path)
    store = PreparedVectorPngStore(path)
    _, options = arguments(original)
    options["image"]["target"] = "media/image1.emf"
    options["region"] = {"x": 1, "y": 2, "width": 4, "height": 5}
    payload, provenance = backend(vector_representations=store).prepare_request(original, **options)
    rep = provenance["representation"]
    assert rep["original_blob_sha256"] == provenance["source"]["original_blob_sha256"] == sha(original)
    assert rep["original_content_type"] == "image/x-emf"
    assert rep["original_width"] is None and rep["original_height"] is None
    assert rep["coordinate_space"] == "prepared-vector-raster-pixels-top-left"
    assert rep["vector_representation"]["conversion"]["payload_image_sha256"] == sha(encoded)
    assert rep["crop"] == options["region"] and rep["raw_bytes_unchanged"] is False
    sent = base64.b64decode(payload["messages"][0]["content"][1]["image_url"]["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(sent)) as image:
        assert image.size == (4, 5)
    assert rep["payload_image_sha256"] == sha(sent)
    assert rep["transform_sha256"] == json_digest({k: v for k, v in rep.items() if k != "transform_sha256"})


@pytest.mark.parametrize("change", ["document", "member", "original", "png", "manifest"])
def test_vector_binding_drift_is_rejected_before_generation(tmp_path, change):
    original, _, path, value = assets(tmp_path)
    store = PreparedVectorPngStore(path)
    _, options = arguments(original)
    options["image"]["target"] = "media/image1.emf"
    model = backend(vector_representations=store)
    if change == "document":
        with pytest.raises(VisionObservationError):
            OwnModelVisionBackend(FakeGeneration(), identity=identity(), deadline=100,
                document_sha256="b" * 64, clock=lambda: 10, vector_representations=store)
        return
    if change == "member":
        options["image"]["target"] = "media/other.emf"
    elif change == "original":
        original = original[:-1] + b"x"
        options["image"]["image_sha256"] = sha(original)
    elif change == "png":
        (tmp_path / "asset.png").write_bytes(png((19, 12)))
    else:
        path.write_text(json.dumps(value) + " ")
    with pytest.raises(VisionObservationError):
        model.prepare_request(original, **options)


def test_cached_observation_input_check_rejects_changed_sidecar(tmp_path):
    original, _, path, _ = assets(tmp_path)
    model = backend(vector_representations=PreparedVectorPngStore(path))
    model.verify_image_inputs(original, member="word/media/image1.emf")
    (tmp_path / "asset.png").write_bytes(png((18, 12)))
    with pytest.raises(VisionObservationError, match="CHANGED"):
        model.verify_image_inputs(original, member="word/media/image1.emf")


@pytest.mark.parametrize("mutation", ["traversal", "symlink", "duplicate-member", "receipt", "pixels", "json-duplicate"])
def test_invalid_representation_stores_fail_closed(tmp_path, mutation):
    _, _, path, value = assets(tmp_path)
    if mutation == "traversal":
        value["entries"][0]["png_path"] = "../asset.png"
    elif mutation == "symlink":
        (tmp_path / "linked.png").symlink_to(tmp_path / "asset.png")
        value["entries"][0]["png_path"] = "linked.png"
    elif mutation == "duplicate-member":
        value["entries"].append(deepcopy(value["entries"][0]))
    elif mutation == "receipt":
        value["entries"][0]["conversion"]["status"] = "verified_fact"
    if mutation == "json-duplicate":
        path.write_text('{"version":"x","version":"y"}')
    else:
        path.write_text(json.dumps(value))
    with pytest.raises(ImageRepresentationError):
        PreparedVectorPngStore(path, max_pixels=1 if mutation == "pixels" else 16_000_000)


@pytest.mark.parametrize("failure", ["unknown", "missing-converter", "pixel-budget", "bad-config"])
def test_conversion_failures_do_not_start_renderer(failure):
    blob, changes = emf(), {}
    if failure == "unknown":
        blob = b"x" * 88
    elif failure == "pixel-budget":
        changes["max_pixels"] = 1
    elif failure == "bad-config":
        changes["dpi"] = True
    with patch("subprocess.run", side_effect=AssertionError("no renderer may run")) as run:
        with pytest.raises(ImageRepresentationError):
            render_vector_png(blob, executable="/missing/soffice", **changes)
    run.assert_not_called()


def test_vector_without_explicit_store_remains_unverified():
    blob = emf()
    _, options = arguments(blob)
    options["image"]["target"] = "media/image1.emf"
    fake = FakeGeneration()
    with pytest.raises(VisionObservationError, match="REPRESENTATION_UNAVAILABLE"):
        backend(fake)(blob, **options)
    assert not fake.calls


def test_runtime_factory_selects_explicit_document_bound_store_and_detects_drift(tmp_path, monkeypatch):
    directory = tmp_path / "images"
    directory.mkdir()
    _, _, manifest, _ = assets(directory)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"service_process": {"cmdline": ["vllm", "--host", "127.0.0.1", "--port", "8000"]}}))
    config = tmp_path / "vision.json"
    config.write_text(json.dumps({"processor_python": "/synthetic/python", "path_audit": str(audit),
        "runtime_environment": {"PATH": "/synthetic/bin", "LD_LIBRARY_PATH": "/synthetic/lib"},
        "image_representation_manifests": [str(manifest)]}))
    monkeypatch.setattr(native, "NativeMultimodalAdmission", StubNativeAdmission)
    with patch("subprocess.run", side_effect=AssertionError("no renderer or model")):
        factory = OwnModelVisionFactory(config, model="cityagent-qwen36-27b-epoch-002",
            model_identity=deepcopy(MODEL_IDENTITY), base_url="http://127.0.0.1:8000")
        actual, _ = factory(generate_observation=lambda *a, **k: None, deadline=999,
                            directory=tmp_path / "work", manifest={"document_sha256": "a" * 64})
        assert isinstance(actual, OwnModelVisionBackend)
        assert actual._vector_representations.document_sha256 == "a" * 64
        assert factory.identity["image_representations"][0]["manifest_sha256"] == sha(manifest.read_bytes())
        (directory / "asset.png").write_bytes(png((19, 12)))
        with pytest.raises(ImageRepresentationError, match="CHANGED"):
            factory.verify()


def test_document_tools_checks_sidecar_inputs_before_cache_hit(tmp_path):
    document, manifest = _image_document(tmp_path)

    class Backend:
        cache_key = "synthetic-binding-check"

        def __init__(self):
            self.checks = self.generations = 0

        def verify_image_inputs(self, blob, *, member):
            self.checks += 1
            assert member.startswith("word/media/")
            if self.checks == 3:
                raise VisionObservationError("VISION_IMAGE_REPRESENTATION_CHANGED", "synthetic drift")

        def __call__(self, *args, **kwargs):
            self.generations += 1
            return {"observation": "synthetic-unverified"}

    fake = Backend()
    tools = DocumentTools(manifest, source_path=document, count_tokens=len, vision_backend=fake)
    first = tools.read_image("I00001", "描述", "{}", [])
    cached = tools.read_image("I00001", "描述", "{}", [])
    changed = tools.read_image("I00001", "描述", "{}", [])
    assert first["ok"] and cached["ok"] and cached["cache"]["judgment_hit"]
    assert not changed["ok"] and fake.checks == 3 and fake.generations == 1


def imported_fodg(blob):
    return ('''<?xml version="1.0" encoding="UTF-8"?>
<office:document xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0">
<office:automatic-styles><style:page-layout style:name="p"><style:page-layout-properties fo:page-width="21cm" fo:page-height="29.7cm"/></style:page-layout></office:automatic-styles>
<office:master-styles><style:master-page style:name="m" style:page-layout-name="p"/></office:master-styles>
<office:body><office:drawing><draw:page draw:master-page-name="m"><draw:frame><draw:image><office:binary-data>%s</office:binary-data></draw:image><draw:image><office:binary-data>%s</office:binary-data></draw:image></draw:frame></draw:page></office:drawing></office:body></office:document>
''' % (base64.b64encode(blob).decode(), base64.b64encode(png()).decode())).encode()


@pytest.mark.parametrize("case", ["success", "white", "transparent", "solid-color", "wrong-size", "no-file", "timeout"])
def test_renderer_receipts_blank_detection_and_failures_are_explicit(tmp_path, case):
    executable = tmp_path / "synthetic-soffice"
    executable.write_bytes(b"synthetic-launcher-not-executed")
    original = emf()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, b"LibreOffice synthetic-contract-test\n", b"")
        directory = Path(argv[argv.index("--outdir") + 1])
        if argv[argv.index("--convert-to") + 1] == "fodg":
            (directory / "original.fodg").write_bytes(imported_fodg(original))
        else:
            if case == "timeout":
                raise subprocess.TimeoutExpired(argv, 1)
            size = (158, 95) if case != "wrong-size" else (157, 95)
            if case != "no-file":
                if case in {"white", "transparent", "solid-color"}:
                    mode, fill = ("RGBA", (0, 123, 45, 0)) if case == "transparent" else ("RGB", "white" if case == "white" else "red")
                    image = Image.new(mode, size, fill)
                    image.save(directory / "bound.png", format="PNG")
                else:
                    (directory / "bound.png").write_bytes(png(size))
        return subprocess.CompletedProcess(argv, 0, b"synthetic-conversion", b"")

    with patch("subprocess.run", side_effect=fake_run):
        if case == "success":
            encoded, receipt = render_vector_png(original, executable=executable)
            assert receipt["output_size"] == [158, 95]
            assert receipt["status"] == "converted_not_semantically_verified"
            assert receipt["payload_image_sha256"] == sha(encoded)
            assert receipt["coordinate_mapping"]["physical_measurements_verified"] is False
        else:
            with pytest.raises(ImageRepresentationError) as error:
                render_vector_png(original, executable=executable)
            if case in {"white", "transparent", "solid-color", "wrong-size"}:
                failure = error.value.failure_receipt
                assert failure["status"] == "failed_unverified"
                assert failure["original_blob_sha256"] == sha(original)
                assert failure["output_sha256"] == sha(error.value.failed_png)
                assert failure["model_generation_attempted"] is False
                assert failure["semantic_quality_passed"] is False
                assert error.value.code == ("IMAGE_CONVERTER_DIMENSIONS_MISMATCH" if case == "wrong-size" else "IMAGE_REPRESENTATION_BLANK")
            else:
                assert error.value.code in {"IMAGE_CONVERTER_FAILED", "IMAGE_CONVERTER_TIMEOUT"}
    assert len(calls) == 3


@pytest.mark.parametrize("fill", ["white", "red", (0, 0, 0, 0)])
def test_blank_prepared_asset_cannot_reach_model_even_with_matching_hashes(tmp_path, fill):
    _, _, path, value = assets(tmp_path)
    image = Image.new("RGBA" if isinstance(fill, tuple) else "RGB", (20, 12), fill)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    encoded = stream.getvalue()
    (tmp_path / "asset.png").write_bytes(encoded)
    receipt = value["entries"][0]["conversion"]
    receipt["payload_image_sha256"], receipt["payload_image_bytes"] = sha(encoded), len(encoded)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = json_digest(receipt)
    path.write_text(json.dumps(value))
    with pytest.raises(ImageRepresentationError, match="BLANK"):
        PreparedVectorPngStore(path)


def test_manifest_replaced_by_symlink_rejects_even_identical_bytes(tmp_path):
    original, _, path, _ = assets(tmp_path)
    store = PreparedVectorPngStore(path)
    identical = tmp_path / "identical.json"
    identical.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(identical)
    with pytest.raises(ImageRepresentationError, match="CHANGED"):
        store.verify()
    with pytest.raises(ImageRepresentationError, match="CHANGED"):
        store.resolve(original, member="word/media/image1.emf", document_sha256="a" * 64)


def test_offline_preparation_keeps_failed_png_and_never_adds_it_to_usable_manifest(tmp_path, monkeypatch):
    import zipfile
    script = ROOT / "scripts/prepare_document_image_representations.py"
    spec = importlib.util.spec_from_file_location("synthetic_preparation_module", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document = tmp_path / "synthetic.docx"
    with zipfile.ZipFile(document, "w") as archive:
        archive.writestr("word/media/image1.emf", emf())
    failed_png = png()
    failure_receipt = {"status": "failed_unverified", "output_sha256": sha(failed_png)}

    def failure(*args, **kwargs):
        raise ImageRepresentationError("IMAGE_REPRESENTATION_BLANK", failure_receipt=failure_receipt, failed_png=failed_png)

    monkeypatch.setattr(module, "render_vector_png", failure)
    output = tmp_path / "prepared"
    report = module.prepare(document, output, executable="/synthetic/soffice", members=["word/media/image1.emf"])
    assert report["preparation_complete"] is False and report["prepared_count"] == 0
    assert json.loads((output / "manifest.json").read_text())["entries"] == []
    failed = report["results"][0]
    assert failed["failed_conversion_receipt"] == failure_receipt
    assert (output / failed["failed_png_path"]).read_bytes() == failed_png
