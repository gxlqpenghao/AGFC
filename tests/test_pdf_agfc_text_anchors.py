import json

from agfc.models import PageAtom
from agfc.text_anchors import (
    FigureAnchorCandidate,
    TextRoleEvidence,
    build_figure_anchor_candidates,
    classify_text_roles,
)


def _text_atom(atom_id: str, text: str, bbox=(72.0, 100.0, 520.0, 116.0)) -> PageAtom:
    return PageAtom(id=atom_id, kind="text_block", bbox=bbox, page_idx=0, text=text)


def test_classify_text_roles_marks_english_figure_caption_prefixes():
    atoms = [
        _text_atom("t1", "Figure 8: local attention score heatmap.", (72.0, 400.0, 520.0, 420.0)),
        _text_atom("t2", "Fig. 4 Ablation overview.", (72.0, 430.0, 520.0, 450.0)),
        _text_atom("t3", "FIGURE 12. System pipeline.", (72.0, 460.0, 520.0, 480.0)),
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.matched_prefix, role.figure_number) for role in roles] == [
        ("t1", "figure_caption", "Figure", "8"),
        ("t2", "figure_caption", "Fig.", "4"),
        ("t3", "figure_caption", "FIGURE", "12"),
    ]
    assert all(role.confidence >= 0.9 for role in roles)


def test_classify_text_roles_marks_chinese_figure_caption_prefixes():
    atoms = [
        _text_atom("zh1", "图 3 总体框架。", (70.0, 360.0, 530.0, 378.0)),
        _text_atom("zh2", "圖4 實驗結果。", (70.0, 390.0, 530.0, 408.0)),
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.matched_prefix, role.figure_number) for role in roles] == [
        ("zh1", "figure_caption", "图", "3"),
        ("zh2", "figure_caption", "圖", "4"),
    ]


def test_classify_text_roles_preserves_hierarchical_chinese_figure_numbers():
    atoms = [
        _text_atom("zh_nested", "图5.2.2-1 LTD-2600 型地质雷达", (90.0, 248.0, 320.0, 263.0)),
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.matched_prefix, role.figure_number) for role in roles] == [
        ("zh_nested", "figure_caption", "图", "5.2.2-1"),
    ]


def test_classify_text_roles_keeps_narrow_sentence_style_caption_prefixes_as_captions():
    atoms = [
        _text_atom("en_caption", "Figure 7 shows the evaluation test rig.", (180.0, 400.0, 420.0, 420.0)),
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.matched_prefix, role.figure_number) for role in roles] == [
        ("en_caption", "figure_caption", "Figure", "7"),
    ]


def test_classify_text_roles_treats_full_width_sentence_initial_figure_reference_as_body_reference():
    atoms = [
        _text_atom("body_lead", "Figure 7 shows the ablation trend across all models.", (72.0, 400.0, 520.0, 420.0)),
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.matched_prefix, role.figure_number) for role in roles] == [
        ("body_lead", "body_reference", "Figure", "7"),
    ]


def test_classify_text_roles_distinguishes_body_references_from_captions():
    atoms = [
        _text_atom("body_en", "The trend is summarized as shown in Figure 2 for all models."),
        _text_atom("body_zh", "如图 3 所示，该模块显著提升了召回率。"),
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.matched_prefix, role.figure_number) for role in roles] == [
        ("body_en", "body_reference", "Figure", "2"),
        ("body_zh", "body_reference", "图", "3"),
    ]
    assert all(role.confidence < 0.9 for role in roles)


def test_classify_text_roles_does_not_promote_merged_caption_and_body_block():
    atoms = [
        _text_atom(
            "merged",
            "Fig. 2: Pipeline overview.\n"
            "3.1. Light field capture and preprocessing\n"
            "The retargeting pipeline starts by capturing real-world LF through several methods.\n"
            "Additional body text continues here with enough length to look like a paragraph.\n"
            "More body text follows in the same extracted text block.",
        )
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.figure_number) for role in roles] == [
        ("merged", "body_reference", "2")
    ]
    assert roles[0].confidence < 0.9


def test_classify_text_roles_marks_short_subfigure_labels():
    atoms = [
        _text_atom("label_a", "(a)", (100.0, 200.0, 112.0, 212.0)),
        _text_atom("label_b", "b)", (150.0, 200.0, 162.0, 212.0)),
        _text_atom("not_label", "(analysis)", (200.0, 200.0, 260.0, 212.0)),
    ]

    roles = classify_text_roles(atoms, page_width=595.0, page_height=842.0)

    assert [(role.atom_id, role.role, role.matched_prefix, role.figure_number) for role in roles] == [
        ("label_a", "subfigure_label", "(a)", None),
        ("label_b", "subfigure_label", "b)", None),
    ]


def test_text_role_evidence_and_anchor_candidates_are_json_serializable():
    evidence = TextRoleEvidence(
        atom_id="t1",
        bbox=(72.0, 400.0, 520.0, 420.0),
        raw_text="Figure 8: local attention score heatmap.",
        role="figure_caption",
        confidence=0.95,
        matched_prefix="Figure",
        figure_number="8",
    )
    anchor = FigureAnchorCandidate(
        id="anchor_t1",
        atom_id="t1",
        bbox=(72.0, 400.0, 520.0, 420.0),
        raw_text="Figure 8: local attention score heatmap.",
        role="figure_caption",
        confidence=0.95,
        matched_prefix="Figure",
        figure_number="8",
    )

    assert evidence.to_dict() == {
        "atom_id": "t1",
        "bbox": [72.0, 400.0, 520.0, 420.0],
        "raw_text": "Figure 8: local attention score heatmap.",
        "role": "figure_caption",
        "confidence": 0.95,
        "matched_prefix": "Figure",
        "figure_number": "8",
    }
    assert anchor.to_dict()["id"] == "anchor_t1"
    json.dumps(evidence.to_dict(), ensure_ascii=False)
    json.dumps(anchor.to_dict(), ensure_ascii=False)


def test_build_figure_anchor_candidates_uses_caption_role_evidence_only():
    roles = classify_text_roles(
        [
            _text_atom("caption", "Fig. 7 Comparison with baselines."),
            _text_atom("reference", "We compare against the baseline in Fig. 7."),
            _text_atom("label", "(a)"),
        ],
        page_width=595.0,
        page_height=842.0,
    )

    anchors = build_figure_anchor_candidates(roles)

    assert [anchor.atom_id for anchor in anchors] == ["caption"]
    assert anchors[0].id == "figure_anchor_caption"
    assert anchors[0].figure_number == "7"
