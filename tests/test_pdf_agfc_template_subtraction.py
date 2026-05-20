from agfc.models import PageAtom
from agfc.template_subtraction import detect_template_atoms


def test_detect_template_atoms_suppresses_repeated_raster_watermark():
    pages = []
    repeated_bbox = (192.84, 262.74, 387.84, 603.66)
    for page_idx in range(8):
        atoms = [
            PageAtom(
                id=f"page_{page_idx}_stamp",
                kind="raster_image",
                bbox=repeated_bbox,
                page_idx=page_idx,
            )
        ]
        pages.append({"page_idx": page_idx, "page_width": 595.0, "page_height": 842.0, "atoms": atoms})

    suppressed, reasons = detect_template_atoms(pages)

    assert all(f"page_{page_idx}_stamp" in suppressed.get(page_idx, set()) for page_idx in range(8))
    assert reasons["page_0_stamp"] == "background_repeat_raster"


def test_detect_template_atoms_suppresses_repeated_vector_background_over_body_text():
    repeated_bbox = (133.0, 250.92, 460.0, 579.92)
    pages = []
    for page_idx in range(6):
        atoms = [
            PageAtom(
                id=f"page_{page_idx}_wm",
                kind="vector_cluster",
                bbox=repeated_bbox,
                page_idx=page_idx,
                metadata={"type": "s", "color": [0.0, 1.0, 0.0], "fill": None, "width": 1.0},
            ),
                PageAtom(
                    id=f"page_{page_idx}_body",
                    kind="text_block",
                    bbox=(150.0, 300.0, 430.0, 360.0),
                    page_idx=page_idx,
                    text="第一行正文内容较长用于模拟段落\n第二行正文内容较长用于模拟段落\n第三行正文内容较长用于模拟段落",
                ),
        ]
        pages.append({"page_idx": page_idx, "page_width": 595.0, "page_height": 842.0, "atoms": atoms})

    suppressed, reasons = detect_template_atoms(pages)

    assert all(f"page_{page_idx}_wm" in suppressed.get(page_idx, set()) for page_idx in range(6))
    assert reasons["page_0_wm"] == "background_repeat_vector"


def test_detect_template_atoms_suppresses_repeated_vector_background_over_two_column_text():
    repeated_bbox = (133.0, 250.92, 460.0, 579.92)
    pages = []
    for page_idx in range(5):
        atoms = [
            PageAtom(
                id=f"page_{page_idx}_wm",
                kind="vector_cluster",
                bbox=repeated_bbox,
                page_idx=page_idx,
                metadata={"type": "s", "color": [0.0, 1.0, 0.0], "fill": None, "width": 1.0},
            ),
            PageAtom(
                id=f"page_{page_idx}_col_left",
                kind="text_block",
                bbox=(160.0, 300.0, 280.0, 430.0),
                page_idx=page_idx,
                text="双栏正文内容较长用于模拟期刊排版\n第二行正文内容较长用于模拟期刊排版",
            ),
            PageAtom(
                id=f"page_{page_idx}_col_right",
                kind="text_block",
                bbox=(315.0, 300.0, 435.0, 430.0),
                page_idx=page_idx,
                text="另一列正文内容较长用于模拟期刊排版\n第二行正文内容较长用于模拟期刊排版",
            ),
        ]
        pages.append({"page_idx": page_idx, "page_width": 595.0, "page_height": 842.0, "atoms": atoms})

    suppressed, reasons = detect_template_atoms(pages)

    assert all(f"page_{page_idx}_wm" in suppressed.get(page_idx, set()) for page_idx in range(5))
    assert reasons["page_0_wm"] == "background_repeat_vector"


def test_detect_template_atoms_suppresses_large_centered_repeated_vector_background():
    repeated_bbox = (133.0, 250.92, 460.0, 579.92)
    pages = []
    for page_idx in range(5):
        atoms = [
            PageAtom(
                id=f"page_{page_idx}_wm_1",
                kind="vector_cluster",
                bbox=repeated_bbox,
                page_idx=page_idx,
                metadata={"type": "s", "color": [0.0, 1.0, 0.0], "fill": None, "width": 1.0},
            ),
            PageAtom(
                id=f"page_{page_idx}_wm_2",
                kind="vector_cluster",
                bbox=repeated_bbox,
                page_idx=page_idx,
                metadata={"type": "s", "color": [0.0, 1.0, 0.0], "fill": None, "width": 1.0},
            ),
            PageAtom(
                id=f"page_{page_idx}_fig",
                kind="raster_image",
                bbox=(70.0, 340.0, 520.0, 620.0),
                page_idx=page_idx,
            ),
        ]
        pages.append({"page_idx": page_idx, "page_width": 595.0, "page_height": 842.0, "atoms": atoms})

    suppressed, reasons = detect_template_atoms(pages)

    assert all(f"page_{page_idx}_wm_1" in suppressed.get(page_idx, set()) for page_idx in range(5))
    assert all(f"page_{page_idx}_wm_2" in suppressed.get(page_idx, set()) for page_idx in range(5))
    assert reasons["page_0_wm_1"] == "background_repeat_vector"


def test_detect_template_atoms_does_not_suppress_panel_border_templates():
    pages = []
    repeated_bbox = (50.0, 100.0, 250.0, 260.0)
    for page_idx in range(6):
        atoms = [
            PageAtom(
                id=f"page_{page_idx}_border",
                kind="panel_border",
                bbox=repeated_bbox,
                page_idx=page_idx,
            )
        ]
        pages.append({"page_idx": page_idx, "page_width": 595.0, "page_height": 842.0, "atoms": atoms})

    suppressed, reasons = detect_template_atoms(pages)

    assert suppressed == {}
    assert reasons == {}
