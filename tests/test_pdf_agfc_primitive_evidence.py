from pathlib import Path

import fitz

from agfc.primitive_evidence import (
    PrimitiveEvidence,
    collect_page_primitive_evidence,
    group_primitive_evidence_clusters,
    select_primitive_evidence_regions,
)


class _FakePrimitivePage:
    def __init__(
        self,
        *,
        cdrawings=None,
        drawings=None,
        blocks=None,
        images=None,
        image_rects=None,
        rect=None,
        fail_on_drawings: bool = False,
    ):
        self._cdrawings = list(cdrawings or [])
        self._drawings = list(drawings or [])
        self._blocks = list(blocks or [])
        self._images = list(images or [])
        self._image_rects = dict(image_rects or {})
        self._fail_on_drawings = fail_on_drawings
        self.rect = rect or fitz.Rect(0.0, 0.0, 600.0, 800.0)

        if cdrawings is None:
            self.get_cdrawings = None  # type: ignore[assignment]

    def get_text(self, mode: str):
        assert mode == "dict"
        return {"blocks": list(self._blocks)}

    def get_cdrawings(self):
        return list(self._cdrawings)

    def get_drawings(self):
        if self._fail_on_drawings:
            raise AssertionError("get_drawings() should not be used when get_cdrawings() is available")
        return list(self._drawings)

    def get_images(self, full: bool = False):
        assert full is True
        return list(self._images)

    def get_image_rects(self, xref: int):
        return list(self._image_rects.get(xref, []))


def test_collect_page_primitive_evidence_prefers_cdrawings_and_splits_fragments():
    page = _FakePrimitivePage(
        cdrawings=[
            {
                "type": "s",
                "rect": (10.0, 20.0, 90.0, 20.0),
                "items": [("l", (10.0, 20.0), (90.0, 20.0))],
                "width": 2.0,
                "color": (0.1, 0.2, 0.3),
                "dashes": "[] 0",
                "seqno": 1,
            },
            {
                "type": "s",
                "rect": (100.0, 100.0, 180.0, 160.0),
                "items": [("re", (100.0, 100.0, 180.0, 160.0), 1)],
                "width": 1.0,
                "color": (0.2, 0.3, 0.4),
                "dashes": "[] 0",
                "seqno": 2,
            },
            {
                "type": "f",
                "rect": (200.0, 220.0, 260.0, 280.0),
                "items": [("re", (200.0, 220.0, 260.0, 280.0), 1)],
                "fill": (0.9, 0.9, 0.7),
                "seqno": 3,
            },
            {
                "type": "s",
                "rect": (300.0, 300.0, 340.0, 340.0),
                "items": [
                    (
                        "c",
                        (300.0, 300.0),
                        (320.0, 280.0),
                        (330.0, 360.0),
                        (340.0, 340.0),
                    )
                ],
                "width": 1.2,
                "color": (0.4, 0.2, 0.1),
                "dashes": "[] 0",
                "seqno": 4,
            },
        ],
        fail_on_drawings=True,
    )

    evidence = collect_page_primitive_evidence(page, page_idx=0)

    kinds = {fragment.kind for fragment in evidence}
    assert {"line", "rect", "curve", "fill"} <= kinds
    assert all(fragment.page_idx == 0 for fragment in evidence)
    assert all(fragment.metadata["source"] == "cdrawings" for fragment in evidence if fragment.kind != "image_anchor")
    assert {fragment.metadata["group_id"] for fragment in evidence if fragment.kind != "image_anchor"} == {
        "page_0_drawing_1",
        "page_0_drawing_2",
        "page_0_drawing_3",
        "page_0_drawing_4",
    }


def test_collect_page_primitive_evidence_falls_back_conservatively_without_cdrawings():
    page = _FakePrimitivePage(
        drawings=[
            {
                "type": "s",
                "rect": fitz.Rect(40.0, 80.0, 140.0, 80.0),
                "items": [("l", fitz.Point(40.0, 80.0), fitz.Point(140.0, 80.0))],
                "width": 1.0,
                "color": (0.2, 0.5, 0.4),
                "dashes": "[] 0",
                "seqno": 10,
            },
            {
                "type": "f",
                "rect": fitz.Rect(160.0, 200.0, 240.0, 280.0),
                "items": [("re", fitz.Rect(160.0, 200.0, 240.0, 280.0), 1)],
                "fill": (0.8, 0.8, 0.6),
                "seqno": 11,
            },
            {
                "type": "f",
                "rect": fitz.Rect(-10.0, -10.0, 610.0, 810.0),
                "items": [("re", fitz.Rect(-10.0, -10.0, 610.0, 810.0), 1)],
                "fill": (0.95, 0.95, 0.95),
                "seqno": 12,
            },
        ],
    )

    evidence = collect_page_primitive_evidence(page, page_idx=0)

    assert [fragment.kind for fragment in evidence] == ["line", "fill"]
    assert evidence[0].bbox == (40.0, 80.0, 140.0, 80.0)
    assert evidence[1].bbox == (160.0, 200.0, 240.0, 280.0)
    assert all(fragment.metadata["source"] == "drawings_fallback" for fragment in evidence)


def test_collect_page_primitive_evidence_adds_deduped_image_anchors():
    page = _FakePrimitivePage(
        cdrawings=[],
        blocks=[{"type": 1, "bbox": (100.0, 200.0, 300.0, 400.0), "number": 3}],
        images=[(17, 0, 0, 0, 0, "", "", "", "", ""), (19, 0, 0, 0, 0, "", "", "", "", "")],
        image_rects={
            17: [fitz.Rect(100.0, 200.0, 300.0, 400.0)],
            19: [fitz.Rect(-10.0, -20.0, 64.0, 820.0)],
        },
        rect=fitz.Rect(0.0, 0.0, 600.0, 800.0),
    )

    evidence = collect_page_primitive_evidence(page, page_idx=0)

    anchors = [fragment for fragment in evidence if fragment.kind == "image_anchor"]
    assert len(anchors) == 2
    assert anchors[0].bbox == (100.0, 200.0, 300.0, 400.0)
    assert anchors[0].metadata["block_number"] == 3
    assert anchors[0].metadata["source"] == "text_dict"
    assert anchors[1].bbox == (0.0, 0.0, 64.0, 800.0)
    assert anchors[1].metadata["xref"] == 19
    assert anchors[1].metadata["source"] == "xref"




def test_group_primitive_evidence_clusters_rolls_up_group_members_by_page():
    evidence = [
        PrimitiveEvidence(
            id="line_1",
            kind="line",
            bbox=(10.0, 10.0, 50.0, 10.0),
            page_idx=0,
            metadata={"group_id": "drawing_a", "source": "cdrawings"},
        ),
        PrimitiveEvidence(
            id="curve_1",
            kind="curve",
            bbox=(20.0, 5.0, 60.0, 40.0),
            page_idx=0,
            metadata={"group_id": "drawing_a", "source": "cdrawings"},
        ),
        PrimitiveEvidence(
            id="fill_1",
            kind="fill",
            bbox=(100.0, 100.0, 150.0, 180.0),
            page_idx=0,
            metadata={"group_id": "drawing_b", "source": "cdrawings"},
        ),
        PrimitiveEvidence(
            id="image_1",
            kind="image_anchor",
            bbox=(25.0, 25.0, 75.0, 90.0),
            page_idx=1,
            metadata={"group_id": "drawing_a", "source": "xref"},
        ),
        PrimitiveEvidence(
            id="rect_1",
            kind="rect",
            bbox=(210.0, 210.0, 240.0, 250.0),
            page_idx=0,
            metadata={},
        ),
    ]

    clusters = group_primitive_evidence_clusters(evidence)

    assert [cluster.id for cluster in clusters] == [
        "page_0_cluster_drawing_a",
        "page_0_cluster_drawing_b",
        "page_1_cluster_drawing_a",
        "page_0_cluster_rect_1",
    ]
    assert clusters[0].group_id == "drawing_a"
    assert clusters[0].bbox == (10.0, 5.0, 60.0, 40.0)
    assert clusters[0].primitive_ids == ("line_1", "curve_1")
    assert clusters[0].kinds == ("curve", "line")
    assert clusters[0].metadata["source"] == "cdrawings"
    assert clusters[2].page_idx == 1
    assert clusters[2].bbox == (25.0, 25.0, 75.0, 90.0)
    assert clusters[3].group_id == "rect_1"
    assert clusters[3].primitive_ids == ("rect_1",)


def test_select_primitive_evidence_regions_expands_to_whole_clusters_and_merges_overlaps():
    evidence = [
        PrimitiveEvidence(
            id="fill_1",
            kind="fill",
            bbox=(10.0, 10.0, 40.0, 40.0),
            page_idx=0,
            metadata={"group_id": "drawing_a"},
        ),
        PrimitiveEvidence(
            id="line_1",
            kind="line",
            bbox=(5.0, 20.0, 35.0, 20.0),
            page_idx=0,
            metadata={"group_id": "drawing_a"},
        ),
        PrimitiveEvidence(
            id="rect_1",
            kind="rect",
            bbox=(35.0, 35.0, 80.0, 80.0),
            page_idx=0,
            metadata={"group_id": "drawing_b"},
        ),
        PrimitiveEvidence(
            id="image_1",
            kind="image_anchor",
            bbox=(150.0, 120.0, 210.0, 200.0),
            page_idx=0,
            metadata={"group_id": "image_a"},
        ),
    ]

    regions = select_primitive_evidence_regions(
        evidence,
        page_idx=0,
        query_bbox=(12.0, 12.0, 55.0, 55.0),
    )

    assert len(regions) == 1
    assert regions[0].bbox == (5.0, 10.0, 80.0, 80.0)
    assert regions[0].cluster_ids == ("page_0_cluster_drawing_a", "page_0_cluster_drawing_b")
    assert regions[0].primitive_ids == ("fill_1", "line_1", "rect_1")
    assert regions[0].kinds == ("fill", "line", "rect")
    assert regions[0].metadata["cluster_count"] == 2
    assert regions[0].metadata["primitive_count"] == 3


def test_select_primitive_evidence_regions_can_filter_by_kind_before_region_building():
    evidence = [
        PrimitiveEvidence(
            id="fill_1",
            kind="fill",
            bbox=(10.0, 10.0, 40.0, 40.0),
            page_idx=0,
            metadata={"group_id": "drawing_a"},
        ),
        PrimitiveEvidence(
            id="rect_1",
            kind="rect",
            bbox=(35.0, 35.0, 80.0, 80.0),
            page_idx=0,
            metadata={"group_id": "drawing_b"},
        ),
        PrimitiveEvidence(
            id="image_1",
            kind="image_anchor",
            bbox=(150.0, 120.0, 210.0, 200.0),
            page_idx=0,
            metadata={"group_id": "image_a"},
        ),
    ]

    regions = select_primitive_evidence_regions(
        evidence,
        page_idx=0,
        query_bbox=(0.0, 0.0, 220.0, 220.0),
        include_kinds={"image_anchor"},
    )

    assert len(regions) == 1
    assert regions[0].bbox == (150.0, 120.0, 210.0, 200.0)
    assert regions[0].cluster_ids == ("page_0_cluster_image_a",)
    assert regions[0].primitive_ids == ("image_1",)


def test_select_primitive_evidence_regions_can_bridge_nearby_flow_like_clusters():
    evidence = [
        PrimitiveEvidence(
            id="rect_1",
            kind="rect",
            bbox=(40.0, 80.0, 92.0, 132.0),
            page_idx=0,
            metadata={"group_id": "flow_a"},
        ),
        PrimitiveEvidence(
            id="line_1",
            kind="line",
            bbox=(92.0, 104.0, 128.0, 104.0),
            page_idx=0,
            metadata={"group_id": "flow_a"},
        ),
        PrimitiveEvidence(
            id="rect_2",
            kind="rect",
            bbox=(138.0, 80.0, 190.0, 132.0),
            page_idx=0,
            metadata={"group_id": "flow_b"},
        ),
    ]

    regions = select_primitive_evidence_regions(
        evidence,
        page_idx=0,
        query_bbox=(30.0, 70.0, 200.0, 140.0),
    )

    assert len(regions) == 1
    assert regions[0].bbox == (40.0, 80.0, 190.0, 132.0)
    assert regions[0].cluster_ids == ("page_0_cluster_flow_a", "page_0_cluster_flow_b")
    assert regions[0].primitive_ids == ("rect_1", "line_1", "rect_2")
