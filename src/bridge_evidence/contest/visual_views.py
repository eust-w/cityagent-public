"""Bounded pixel views for a small, explicit subset of DrawingML transforms.

These are source-pixel views, not a Word page renderer. Every source pixel whose
area intersects the visible crop is retained; no resize or interpolation occurs.
The exact OOXML crop and the intersecting integer pixel box are both recorded.
"""

from __future__ import annotations

import hashlib
import io
import re

from .visual_inventory import A

VIEW_VERSION = "ooxml-visible-raster-v1"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def _integer(value, minimum, maximum):
    if not isinstance(value, str) or not re.fullmatch(r"[+-]?[0-9]{1,14}", value):
        raise ValueError("invalid integer picture transform")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError("picture transform outside supported range")
    return result


def _leaf(node, attributes):
    if (
        set(node.attrib) - attributes
        or len(node)
        or (node.text or "").strip()
        or (node.tail or "").strip()
    ):
        raise ValueError("unsupported picture transform structure")


def picture_transform(blip):
    """Parse only direct picture crop and shape transforms, never group effects."""
    picture = next((p for p in blip.iterancestors() if p.tag == f"{{{PIC}}}pic"), None)
    if picture is None:
        raise ValueError("picture transform requires a bound pic element")
    if any(
        child.tag not in {f"{{{PIC}}}nvPicPr", f"{{{PIC}}}blipFill", f"{{{PIC}}}spPr"}
        for child in picture
    ):
        raise ValueError("unsupported picture styling")
    fill = blip.getparent()
    if fill.tag != f"{{{PIC}}}blipFill" or fill.getparent() is not picture:
        raise ValueError("ambiguous picture fill")
    if set(fill.attrib) - {"dpi", "rotWithShape"}:
        raise ValueError("unsupported picture fill attributes")
    if fill.get("rotWithShape", "1") not in {"1", "true"}:
        raise ValueError("unrotated fill in rotated shape is not supported")
    if "dpi" in fill.attrib:
        _integer(fill.get("dpi"), 0, 4294967295)
    if len(picture.findall(f"{{{PIC}}}blipFill")) != 1:
        raise ValueError("ambiguous picture fill")
    if any(
        child.tag not in {f"{{{A}}}blip", f"{{{A}}}srcRect", f"{{{A}}}stretch"}
        for child in fill
    ):
        raise ValueError("unsupported picture fill operation")
    if len(fill.findall(f"{{{A}}}blip")) != 1:
        raise ValueError("ambiguous picture image")
    crops = fill.findall(f"{{{A}}}srcRect")
    stretches = fill.findall(f"{{{A}}}stretch")
    if len(crops) > 1 or len(stretches) != 1:
        raise ValueError("ambiguous picture crop or stretch")
    stretch = stretches[0]
    if (
        stretch.attrib
        or (stretch.text or "").strip()
        or len(stretch) > 1
        or (len(stretch) and stretch[0].tag != f"{{{A}}}fillRect")
    ):
        raise ValueError("unsupported picture stretch")
    if len(stretch):
        _leaf(stretch[0], {"l", "t", "r", "b"})
        if any(_integer(v, 0, 100000) for v in stretch[0].attrib.values()):
            raise ValueError("nonzero fill rectangle is not supported")
    crop = {key: 0 for key in ("l", "t", "r", "b")}
    if crops:
        _leaf(crops[0], set(crop))
        crop.update(
            {key: _integer(value, 0, 100000) for key, value in crops[0].attrib.items()}
        )
    if crop["l"] + crop["r"] >= 100000 or crop["t"] + crop["b"] >= 100000:
        raise ValueError("picture crop has no positive visible area")
    shapes = picture.findall(f"{{{PIC}}}spPr")
    if len(shapes) != 1:
        raise ValueError("ambiguous picture shape")
    shape = shapes[0]
    if set(shape.attrib) - {"bwMode"} or shape.get("bwMode", "auto") not in {
        "auto",
        "clr",
    }:
        raise ValueError("unsupported picture shape attributes")
    if any(
        child.tag
        not in {f"{{{A}}}xfrm", f"{{{A}}}prstGeom", f"{{{A}}}ln", f"{{{A}}}noFill"}
        for child in shape
    ):
        raise ValueError("unsupported picture shape operation")
    fills = shape.findall(f"{{{A}}}noFill")
    if len(fills) > 1:
        raise ValueError("ambiguous picture shape fill")
    for fill in fills:
        _leaf(fill, set())
    geometries = shape.findall(f"{{{A}}}prstGeom")
    if len(geometries) != 1 or len(shape.findall(f"{{{A}}}ln")) > 1:
        raise ValueError("ambiguous picture geometry")
    geometry = geometries[0]
    if (
        dict(geometry.attrib) != {"prst": "rect"}
        or len(geometry) > 1
        or (geometry.text or "").strip()
    ):
        raise ValueError("unsupported picture geometry")
    if len(geometry):
        if geometry[0].tag != f"{{{A}}}avLst":
            raise ValueError("unsupported picture geometry child")
        _leaf(geometry[0], set())
    transforms = shape.findall(f"{{{A}}}xfrm")
    if len(transforms) != 1:
        raise ValueError("ambiguous picture transform")
    xfrm = transforms[0]
    if set(xfrm.attrib) - {"rot", "flipH", "flipV"} or (xfrm.text or "").strip():
        raise ValueError("unsupported picture transform attributes")
    rotation = _integer(xfrm.get("rot", "0"), -21600000, 21600000)
    if rotation % 5400000:
        raise ValueError("only quarter-turn picture rotations are supported")
    flips = {}
    for key in ("flipH", "flipV"):
        value = xfrm.get(key, "0")
        if value not in {"0", "1", "true", "false"}:
            raise ValueError("invalid picture flip")
        flips[key] = value in {"1", "true"}
    if [child.tag for child in xfrm] != [f"{{{A}}}off", f"{{{A}}}ext"]:
        raise ValueError("unsupported picture transform children")
    for child, names, minimum in (
        (xfrm[0], {"x", "y"}, -27273042329600),
        (xfrm[1], {"cx", "cy"}, 1),
    ):
        _leaf(child, names)
        if set(child.attrib) != names:
            raise ValueError("incomplete picture transform geometry")
        for value in child.attrib.values():
            _integer(value, minimum, 27273042329600)
    return {
        "crop_100000": crop,
        "clockwise_quarter_turns": (rotation // 5400000) % 4,
        **flips,
    }


def transformed_view(data, original, transform, config):
    """Return lossless pixel-view bytes and an exact pixel-edge inverse map."""
    from PIL import Image

    if not (
        any(transform["crop_100000"].values())
        or transform["clockwise_quarter_turns"]
        or transform["flipH"]
        or transform["flipV"]
    ):
        return data, original
    width, height = original["width"], original["height"]
    crop = transform["crop_100000"]
    # Outward rounding retains every partially visible boundary pixel. Keep the
    # exact rational crop alongside this box; never claim subpixel page rendering.
    box = [
        width * crop["l"] // 100000,
        height * crop["t"] // 100000,
        (width * (100000 - crop["r"]) + 99999) // 100000,
        (height * (100000 - crop["b"]) + 99999) // 100000,
    ]
    left, top, right, bottom = box
    cw, ch = right - left, bottom - top
    with Image.open(io.BytesIO(data)) as source:
        raster = source.convert("RGBA").crop(box)
    # DrawingML flips the local shape axes before rotating the whole shape.
    if transform["flipH"]:
        raster = raster.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if transform["flipV"]:
        raster = raster.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    turns = transform["clockwise_quarter_turns"]
    if turns:
        raster = raster.transpose(
            {
                1: Image.Transpose.ROTATE_270,
                2: Image.Transpose.ROTATE_180,
                3: Image.Transpose.ROTATE_90,
            }[turns]
        )

    def inverse(x, y):
        u, v = [(x, y), (y, ch - x), (cw - x, ch - y), (cw - y, x)][turns]
        if transform["flipH"]:
            u = cw - u
        if transform["flipV"]:
            v = ch - v
        return u + left, v + top

    origin, ex, ey = inverse(0, 0), inverse(1, 0), inverse(0, 1)
    matrix = [
        ex[0] - origin[0],
        ey[0] - origin[0],
        origin[0],
        ex[1] - origin[1],
        ey[1] - origin[1],
        origin[1],
    ]
    output = io.BytesIO()
    raster.save(output, format="PNG", optimize=False, compress_level=6)
    encoded = output.getvalue()
    if len(encoded) > config["max_image_bytes"]:
        raise ValueError("transformed image exceeds encoded byte limit")
    return encoded, {
        "recipe": VIEW_VERSION,
        "mime_type": "image/png",
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
        "width": raster.width,
        "height": raster.height,
        "mode": "RGBA",
        "original_encoded_sha256": original["encoded_sha256"],
        "original_size": [width, height],
        "transform": transform,
        "source_pixel_box": box,
        "crop_rasterization": "all-intersecting-source-pixels-v1",
        "view_to_original_pixel_edges": matrix,
    }


def original_bbox(view, bbox):
    """Map normalized view bounds to original normalized source-pixel bounds."""
    matrix = view["view_to_original_pixel_edges"]
    width, height = view["original_size"]
    a, b, c, d, e, f = matrix
    points = [
        (
            a * x * view["width"] + b * y * view["height"] + c,
            d * x * view["width"] + e * y * view["height"] + f,
        )
        for x in (bbox[0], bbox[2])
        for y in (bbox[1], bbox[3])
    ]
    return [
        min(p[0] for p in points) / width,
        min(p[1] for p in points) / height,
        max(p[0] for p in points) / width,
        max(p[1] for p in points) / height,
    ]
