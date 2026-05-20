from pathlib import Path

import fitz

from agfc.atoms import collect_page_atoms


def _build_sample_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(50, 100, 250, 260))
    shape.finish(
        color=(0.2, 0.4, 0.8),
        width=1.5,
        dashes="[6 4] 0",
    )
    shape.draw_rect(fitz.Rect(60, 225, 240, 255))
    shape.finish(fill=(0.8, 0.9, 0.6), color=None)
    shape.commit()
    page.insert_textbox(fitz.Rect(70, 130, 220, 170), "Panel Title", fontsize=14)
    page.insert_textbox(fitz.Rect(50, 320, 400, 380), "This is body text outside the panel.", fontsize=12)
    doc.save(path)
    doc.close()


def test_collect_page_atoms_extracts_text_border_and_color_band(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    _build_sample_pdf(pdf_path)

    doc = fitz.open(pdf_path)
    atoms = collect_page_atoms(doc[0], page_idx=0)
    doc.close()

    kinds = [atom.kind for atom in atoms]
    assert "text_block" in kinds
    assert "panel_border" in kinds
    assert "color_band" in kinds

    title_atoms = [atom for atom in atoms if atom.kind == "text_block" and "Panel Title" in atom.text]
    assert len(title_atoms) == 1


def test_collect_page_atoms_extracts_slender_chart_bars_as_vector_clusters(tmp_path: Path):
    pdf_path = tmp_path / "bars.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(80, 220, 90, 320))
    shape.finish(fill=(0.1, 0.5, 0.8), color=None)
    shape.draw_rect(fitz.Rect(100, 200, 110, 320))
    shape.finish(fill=(0.2, 0.6, 0.9), color=None)
    shape.commit()
    doc.save(pdf_path)
    doc.close()

    doc = fitz.open(pdf_path)
    atoms = collect_page_atoms(doc[0], page_idx=0)
    doc.close()

    vector_atoms = [atom for atom in atoms if atom.kind == "vector_cluster"]
    assert len(vector_atoms) >= 2


def test_collect_page_atoms_aggregates_thin_drawing_groups_into_composite_vector_cluster(tmp_path: Path):
    pdf_path = tmp_path / "thin_group.pdf"
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)

    shape = page.new_shape()
    for y in (220, 250, 280, 310, 340):
        shape.draw_line(fitz.Point(120, y), fitz.Point(280, y))
        shape.finish(color=(0.2, 0.4, 0.8), width=1.0)
    for x in (140, 190, 240):
        shape.draw_line(fitz.Point(x, 210), fitz.Point(x, 350))
        shape.finish(color=(0.2, 0.4, 0.8), width=1.0)
    shape.commit()

    doc.save(pdf_path)
    doc.close()

    doc = fitz.open(pdf_path)
    atoms = collect_page_atoms(doc[0], page_idx=0)
    doc.close()

    composite_vectors = [
        atom
        for atom in atoms
        if atom.kind == "vector_cluster" and atom.metadata.get("composite_member_count", 0) >= 3
    ]
    assert len(composite_vectors) >= 1
    bbox = composite_vectors[0].bbox
    assert bbox[0] <= 140.0
    assert bbox[1] <= 220.0
    assert bbox[2] >= 260.0
    assert bbox[3] >= 340.0


class _FakeXrefImagePage:
    def __init__(self, *, blocks, images, image_rects, drawings=None, rect=None):
        self._blocks = blocks
        self._images = images
        self._image_rects = image_rects
        self._drawings = drawings or []
        self.rect = rect or fitz.Rect(0.0, 0.0, 600.0, 800.0)

    def get_text(self, mode: str):
        assert mode == "dict"
        return {"blocks": list(self._blocks)}

    def get_images(self, full: bool = False):
        assert full is True
        return list(self._images)

    def get_image_rects(self, xref: int):
        return list(self._image_rects.get(xref, []))

    def get_drawings(self):
        return list(self._drawings)


def test_collect_page_atoms_recovers_xref_only_images():
    page = _FakeXrefImagePage(
        blocks=[],
        images=[(17, 0, 0, 0, 0, "", "", "", "", "")],
        image_rects={17: [fitz.Rect(100.0, 200.0, 300.0, 400.0)]},
    )

    atoms = collect_page_atoms(page, page_idx=0)

    raster_atoms = [atom for atom in atoms if atom.kind == "raster_image"]
    assert len(raster_atoms) == 1
    assert raster_atoms[0].bbox == (100.0, 200.0, 300.0, 400.0)
    assert raster_atoms[0].metadata["xref"] == 17


def test_collect_page_atoms_dedupes_xref_images_overlapping_text_blocks():
    page = _FakeXrefImagePage(
        blocks=[{"type": 1, "bbox": (100.0, 200.0, 300.0, 400.0), "number": 3}],
        images=[(17, 0, 0, 0, 0, "", "", "", "", "")],
        image_rects={17: [fitz.Rect(100.0, 200.0, 300.0, 400.0)]},
    )

    atoms = collect_page_atoms(page, page_idx=0)

    raster_atoms = [atom for atom in atoms if atom.kind == "raster_image"]
    assert len(raster_atoms) == 1
    assert raster_atoms[0].metadata["block_number"] == 3


def test_collect_page_atoms_clips_xref_image_bboxes_to_page_bounds():
    page = _FakeXrefImagePage(
        blocks=[],
        images=[(17, 0, 0, 0, 0, "", "", "", "", "")],
        image_rects={17: [fitz.Rect(-10.0, -20.0, 610.0, 820.0)]},
        rect=fitz.Rect(0.0, 0.0, 600.0, 800.0),
    )

    atoms = collect_page_atoms(page, page_idx=0)

    raster_atoms = [atom for atom in atoms if atom.kind == "raster_image"]
    assert len(raster_atoms) == 1
    assert raster_atoms[0].bbox == (0.0, 0.0, 600.0, 800.0)


def test_collect_page_atoms_preserves_edge_spanning_skinny_xref_images_as_atoms():
    page = _FakeXrefImagePage(
        blocks=[],
        images=[(17, 0, 0, 0, 0, "", "", "", "", "")],
        image_rects={17: [fitz.Rect(-8.0, -8.0, 64.0, 808.0)]},
        rect=fitz.Rect(0.0, 0.0, 600.0, 800.0),
    )

    atoms = collect_page_atoms(page, page_idx=0)

    raster_atoms = [atom for atom in atoms if atom.kind == "raster_image"]
    assert len(raster_atoms) == 1
    assert raster_atoms[0].bbox == (0.0, 0.0, 64.0, 800.0)


def test_collect_page_atoms_adds_conservative_drawing_composites_without_overmerging():
    page = _FakeXrefImagePage(
        blocks=[],
        images=[],
        image_rects={},
        drawings=[
            {"type": "f", "rect": fitz.Rect(160.0, 120.0, 210.0, 240.0), "fill": (0.9, 0.9, 0.9)},
            {"type": "s", "rect": fitz.Rect(40.0, 120.0, 260.0, 120.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(40.0, 160.0, 260.0, 160.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(40.0, 200.0, 260.0, 200.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(40.0, 240.0, 260.0, 240.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(210.0, 120.0, 210.0, 240.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(260.0, 120.0, 260.0, 240.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "f", "rect": fitz.Rect(160.0, 430.0, 210.0, 570.0), "fill": (0.9, 0.9, 0.9)},
            {"type": "s", "rect": fitz.Rect(40.0, 430.0, 260.0, 430.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(40.0, 470.0, 260.0, 470.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(40.0, 520.0, 260.0, 520.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(40.0, 570.0, 260.0, 570.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(210.0, 430.0, 210.0, 570.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
            {"type": "s", "rect": fitz.Rect(260.0, 430.0, 260.0, 570.0), "color": (0.2, 0.5, 0.4), "dashes": "[] 0", "width": 1.0},
        ],
    )

    atoms = collect_page_atoms(page, page_idx=0)

    composite_atoms = [
        atom
        for atom in atoms
        if atom.kind == "vector_cluster" and atom.metadata.get("source") == "drawing_composite"
    ]
    assert len(composite_atoms) == 2
    assert composite_atoms[0].bbox == (40.0, 120.0, 260.0, 240.0)
    assert composite_atoms[1].bbox == (40.0, 430.0, 260.0, 570.0)
    assert all(atom.metadata["member_count"] >= 6 for atom in composite_atoms)
