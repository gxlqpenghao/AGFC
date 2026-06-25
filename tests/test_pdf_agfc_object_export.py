from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image

from agfc.object_export import extract_clean_figure_image
from agfc.object_export import find_matching_image_xobject


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakePage:
    def __init__(self, *, image_entries: list[tuple], image_rects: dict[int, list[fitz.Rect]]):
        self._image_entries = image_entries
        self._image_rects = image_rects

    def get_images(self, full: bool = False):
        assert full is True
        return list(self._image_entries)

    def get_image_rects(self, xref: int):
        return list(self._image_rects.get(xref, []))


class FakeDoc:
    def __init__(self, payloads: dict[int, dict[str, object]]):
        self._payloads = payloads

    def extract_image(self, xref: int):
        return dict(self._payloads[xref])


def test_find_matching_image_xobject_prefers_localized_xref_over_repeated_template():
    page = FakePage(
        image_entries=[
            (140, 0, 1200, 1600, 8, "DeviceRGB", "", "Im140", "FlateDecode", 0),
            (200, 301, 500, 320, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={
            140: [fitz.Rect(0.0, 0.0, 612.0, 792.0)],
            200: [fitz.Rect(98.0, 202.0, 402.0, 518.0)],
        },
    )

    match = find_matching_image_xobject(
        page,
        figure_bbox=(100.0, 200.0, 400.0, 520.0),
        xref_usage_counts={140: 45, 200: 1},
    )

    assert match is not None
    assert match.xref == 200
    assert match.smask == 301
    assert match.bbox == (98.0, 202.0, 402.0, 518.0)
    assert match.usage_count == 1


def test_extract_clean_figure_image_applies_smask_alpha():
    page = FakePage(
        image_entries=[
            (200, 301, 2, 1, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={200: [fitz.Rect(100.0, 200.0, 200.0, 250.0)]},
    )
    doc = FakeDoc(
        {
            200: {
                "image": _png_bytes(Image.new("RGB", (2, 1), (255, 0, 0))),
                "ext": "png",
            },
            301: {
                "image": _png_bytes(Image.frombytes("L", (2, 1), bytes([0, 255]))),
                "ext": "png",
            },
        }
    )

    image = extract_clean_figure_image(
        doc,
        page,
        figure_bbox=(100.0, 200.0, 200.0, 250.0),
        xref_usage_counts={200: 1},
    )

    assert image is not None
    assert image.mode == "RGBA"
    assert image.size == (2, 1)
    assert image.getpixel((0, 0)) == (255, 0, 0, 0)
    assert image.getpixel((1, 0)) == (255, 0, 0, 255)


def test_find_matching_image_xobject_rejects_geometric_mismatch():
    page = FakePage(
        image_entries=[
            (140, 0, 1200, 1600, 8, "DeviceRGB", "", "Im140", "FlateDecode", 0),
            (201, 0, 320, 180, 8, "DeviceRGB", "", "Im201", "FlateDecode", 0),
        ],
        image_rects={
            140: [fitz.Rect(0.0, 0.0, 612.0, 792.0)],
            201: [fitz.Rect(420.0, 520.0, 560.0, 640.0)],
        },
    )

    match = find_matching_image_xobject(
        page,
        figure_bbox=(100.0, 200.0, 240.0, 320.0),
        xref_usage_counts={140: 45, 201: 1},
    )

    assert match is None


def test_extract_clean_figure_image_can_save_png(tmp_path: Path):
    page = FakePage(
        image_entries=[
            (200, 0, 2, 2, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={200: [fitz.Rect(100.0, 200.0, 200.0, 300.0)]},
    )
    doc = FakeDoc(
        {
            200: {
                "image": _png_bytes(Image.new("RGB", (2, 2), (0, 128, 255))),
                "ext": "png",
            },
        }
    )
    output_path = tmp_path / "figure.png"

    image = extract_clean_figure_image(
        doc,
        page,
        figure_bbox=(100.0, 200.0, 200.0, 300.0),
        xref_usage_counts={200: 1},
        output_path=output_path,
    )

    assert image is not None
    assert output_path.exists()
    with Image.open(output_path) as saved:
        assert saved.mode == "RGB"
        assert saved.size == (2, 2)


def test_extract_clean_figure_image_requires_usage_counts():
    page = FakePage(
        image_entries=[
            (200, 0, 2, 2, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={200: [fitz.Rect(100.0, 200.0, 200.0, 300.0)]},
    )
    doc = FakeDoc(
        {
            200: {
                "image": _png_bytes(Image.new("RGB", (2, 2), (0, 128, 255))),
                "ext": "png",
            },
        }
    )

    try:
        extract_clean_figure_image(
            doc,
            page,
            figure_bbox=(100.0, 200.0, 200.0, 300.0),
            xref_usage_counts=None,
        )
    except ValueError as exc:
        assert "xref_usage_counts" in str(exc)
    else:
        raise AssertionError("expected ValueError when xref_usage_counts is omitted")


def test_extract_clean_figure_image_rejects_non_png_output_suffix(tmp_path: Path):
    page = FakePage(
        image_entries=[
            (200, 0, 2, 2, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={200: [fitz.Rect(100.0, 200.0, 200.0, 300.0)]},
    )
    doc = FakeDoc(
        {
            200: {
                "image": _png_bytes(Image.new("RGB", (2, 2), (0, 128, 255))),
                "ext": "png",
            },
        }
    )

    try:
        extract_clean_figure_image(
            doc,
            page,
            figure_bbox=(100.0, 200.0, 200.0, 300.0),
            xref_usage_counts={200: 1},
            output_path=tmp_path / "figure.jpg",
        )
    except ValueError as exc:
        assert ".png" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-png output path")


def test_extract_clean_figure_image_flattens_alpha_when_saving_png(tmp_path: Path):
    page = FakePage(
        image_entries=[
            (200, 301, 2, 1, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={200: [fitz.Rect(100.0, 200.0, 200.0, 250.0)]},
    )
    doc = FakeDoc(
        {
            200: {
                "image": _png_bytes(Image.new("RGB", (2, 1), (255, 0, 0))),
                "ext": "png",
            },
            301: {
                "image": _png_bytes(Image.frombytes("L", (2, 1), bytes([0, 255]))),
                "ext": "png",
            },
        }
    )
    output_path = tmp_path / "figure.png"

    extract_clean_figure_image(
        doc,
        page,
        figure_bbox=(100.0, 200.0, 200.0, 250.0),
        xref_usage_counts={200: 1},
        output_path=output_path,
    )

    with Image.open(output_path) as saved:
        assert saved.mode == "RGB"
        assert saved.getpixel((0, 0)) == (255, 255, 255)
        assert saved.getpixel((1, 0)) == (255, 0, 0)


def test_extract_clean_figure_image_returns_none_when_primary_image_decode_fails():
    page = FakePage(
        image_entries=[
            (200, 0, 2, 2, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={200: [fitz.Rect(100.0, 200.0, 200.0, 300.0)]},
    )
    doc = FakeDoc(
        {
            200: {
                "image": b"not-a-real-image",
                "ext": "png",
            },
        }
    )

    image = extract_clean_figure_image(
        doc,
        page,
        figure_bbox=(100.0, 200.0, 200.0, 300.0),
        xref_usage_counts={200: 1},
    )

    assert image is None


def test_extract_clean_figure_image_ignores_invalid_smask_and_keeps_base_image():
    page = FakePage(
        image_entries=[
            (200, 301, 2, 1, 8, "DeviceRGB", "", "Im200", "FlateDecode", 0),
        ],
        image_rects={200: [fitz.Rect(100.0, 200.0, 200.0, 250.0)]},
    )
    doc = FakeDoc(
        {
            200: {
                "image": _png_bytes(Image.new("RGB", (2, 1), (255, 0, 0))),
                "ext": "png",
            },
            301: {
                "image": b"not-a-real-mask",
                "ext": "png",
            },
        }
    )

    image = extract_clean_figure_image(
        doc,
        page,
        figure_bbox=(100.0, 200.0, 200.0, 250.0),
        xref_usage_counts={200: 1},
    )

    assert image is not None
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((1, 0)) == (255, 0, 0)
