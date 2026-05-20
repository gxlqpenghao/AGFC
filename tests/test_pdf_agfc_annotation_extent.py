from agfc.annotation_extent import is_nonraster_annotation_atom, looks_like_table_cell_text
from agfc.models import PageAtom


def test_axis_tick_block_is_visual_annotation_not_table_cell():
    atom = PageAtom(
        id="x_axis",
        kind="text_block",
        bbox=(80.0, 244.0, 305.0, 264.0),
        page_idx=0,
        text="0\n2\n4\n6\n8\n10\nTarget spectral efficiency [bits/sec/Hz]",
    )

    assert looks_like_table_cell_text(atom.text) is False
    assert is_nonraster_annotation_atom(atom, content_bbox=(70.0, 80.0, 305.0, 242.0)) is True


def test_domain_word_sequence_is_not_visual_annotation_without_structure():
    atom = PageAtom(
        id="domain_words",
        kind="text_block",
        bbox=(80.0, 244.0, 305.0, 264.0),
        page_idx=0,
        text="target spectral throughput response recall values layer head",
    )

    assert is_nonraster_annotation_atom(atom, content_bbox=(70.0, 80.0, 305.0, 242.0)) is False


def test_multiline_chart_legend_is_visual_annotation():
    atom = PageAtom(
        id="legend",
        kind="text_block",
        bbox=(95.0, 64.0, 174.0, 102.0),
        page_idx=0,
        text="LoS Concentration\nUniform Path Allocation\nOutMin w/ Average SE\nOutMin",
    )

    assert is_nonraster_annotation_atom(atom, content_bbox=(70.0, 82.0, 305.0, 242.0)) is True


def test_short_multiline_legend_list_is_visual_annotation():
    atom = PageAtom(
        id="legend_list",
        kind="text_block",
        bbox=(470.0, 103.0, 499.0, 140.0),
        page_idx=0,
        text="statement\nclarification\ninform\nyes-no-question\nthanking\nclosing\nsuggestion",
    )

    assert is_nonraster_annotation_atom(atom, content_bbox=(340.0, 72.0, 500.0, 171.0)) is True


def test_section_heading_below_visual_is_not_annotation():
    atom = PageAtom(
        id="section",
        kind="text_block",
        bbox=(50.0, 380.0, 175.0, 391.0),
        page_idx=0,
        text="A.3. SynthText in the Wild",
    )

    assert is_nonraster_annotation_atom(atom, content_bbox=(50.0, 126.0, 545.0, 356.0)) is False


def test_model_metric_table_text_stays_table_like():
    assert looks_like_table_cell_text("Model Acc F1 0.81 0.77 0.84 0.79") is True


def test_chinese_form_row_text_stays_table_like_not_annotation():
    text = "1 国家自然科学基金重点项目：超大跨 CFST 拱桥整体性能设计理论与施工控制方法"
    atom = PageAtom(
        id="form_row",
        kind="text_block",
        bbox=(88.0, 620.0, 530.0, 660.0),
        page_idx=0,
        text=text,
    )

    assert looks_like_table_cell_text(text) is True
    assert is_nonraster_annotation_atom(atom, content_bbox=(60.0, 300.0, 535.0, 610.0)) is False


def test_chinese_body_heading_near_raster_is_not_annotation():
    atom = PageAtom(
        id="body_heading",
        kind="text_block",
        bbox=(66.0, 314.0, 530.0, 338.0),
        page_idx=0,
        text="（2）“钢管混凝土拱桥主拱管内混凝土密实状态多因素定量评估技术”",
    )

    assert is_nonraster_annotation_atom(atom, content_bbox=(90.0, 70.0, 506.0, 310.0)) is False


def test_page_number_footer_is_not_annotation():
    atom = PageAtom(
        id="footer",
        kind="text_block",
        bbox=(285.0, 774.0, 308.0, 790.0),
        page_idx=0,
        text="6 / 28",
    )

    assert is_nonraster_annotation_atom(atom, content_bbox=(101.0, 298.0, 495.0, 744.0)) is False


def test_chinese_numbered_paragraph_after_caption_is_not_chained_annotation():
    atom = PageAtom(
        id="next_paragraph",
        kind="text_block",
        bbox=(90.0, 679.0, 529.0, 697.0),
        page_idx=0,
        text="（3）“拱桥施工期数字孪生管控平台”应用在了平陆运河G75 钦江大桥、南珠大",
    )

    assert is_nonraster_annotation_atom(atom, content_bbox=(95.0, 527.0, 501.0, 671.0)) is False
