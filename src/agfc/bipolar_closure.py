from __future__ import annotations

from agfc.pipeline_models import BBox, BipolarEdge, ClosureResult


_SUPPORT_ONLY_RELATIONS = {"caption_of"}


def compute_bipolar_closure(
    *,
    seed_id: str,
    edges: list[BipolarEdge],
    node_bboxes: dict[str, BBox] | None = None,
    atom_ids: set[str] | None = None,
    seed_bbox: BBox | None = None,
) -> ClosureResult:
    node_bboxes = node_bboxes or {}
    atom_ids = atom_ids or set()
    adjacency: dict[str, list[BipolarEdge]] = {}
    blocked_targets_by_source: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_id, []).append(edge)
        if edge.polarity == "repel":
            blocked_targets_by_source.setdefault(edge.source_id, set()).add(edge.target_id)

    queue = [seed_id]
    seen: set[str] = set()
    support_only_nodes: set[str] = set()
    while queue:
        current = queue.pop(0)
        for edge in adjacency.get(current, []):
            if edge.polarity != "attract":
                continue
            if edge.target_id in blocked_targets_by_source.get(current, set()):
                continue
            is_support_only = edge.relation in _SUPPORT_ONLY_RELATIONS
            if edge.target_id in seen:
                if not is_support_only:
                    support_only_nodes.discard(edge.target_id)
                continue
            seen.add(edge.target_id)
            if is_support_only:
                support_only_nodes.add(edge.target_id)
            queue.append(edge.target_id)

    content_nodes = seen - support_only_nodes
    return ClosureResult(
        seed_id=seed_id,
        node_ids=sorted(seen),
        atom_ids=sorted(node_id for node_id in content_nodes if node_id in atom_ids),
        bbox=_union_bbox(
            [node_bboxes[node_id] for node_id in content_nodes if node_id in node_bboxes]
            + ([seed_bbox] if seed_bbox is not None else [])
        ),
        level="L2",
    )


def _union_bbox(bboxes: list[BBox]) -> BBox:
    if not bboxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0, y0, x1, y1 = bboxes[0]
    for bx0, by0, bx1, by1 in bboxes[1:]:
        x0 = min(x0, bx0)
        y0 = min(y0, by0)
        x1 = max(x1, bx1)
        y1 = max(y1, by1)
    return (x0, y0, x1, y1)
