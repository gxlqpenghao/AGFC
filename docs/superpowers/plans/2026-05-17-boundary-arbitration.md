# Boundary Arbitration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace distributed boundary authority with one final `BoundaryProposal` / `BoundaryArbitration` layer that fixes v44's 13 hidden quality-fail matches without reintroducing fallback-driven overfitting.

**Architecture:** Existing modules should emit evidence and proposals; only a shared boundary arbitrator selects the exported bbox. To minimize churn, define `BoundaryProposal` and arbitration helpers inside the existing `boundary.py` first, keep `BoundaryCalibrationResult` compatible, and only split a separate module later if the file becomes too large. Raster groups, complete objects, annotation extent, primitive/pixel refinement, and raster split become comparable proposals with explicit reject reasons.

**Tech Stack:** Python, existing AGFC modules under `src/agfc`, pytest, JournalMix v1 benchmark artifacts.

---

## File Structure

- Modify: `src/agfc/boundary.py`
  - Add `BoundaryProposal` / `BoundaryDecision` structures and arbitration helpers.
  - Split candidate generation from `max()` selection.
  - Keep `BoundaryCalibrationResult` and existing metadata compatible during migration.
- Modify: `src/agfc/figure_instances.py`
  - Build proposal pool and call the arbitrator once.
  - Stop using raster visual group bbox as mandatory boundary floor.
  - Stop skipping annotation evidence when a raster group exists.
- Modify: `src/agfc/annotation_extent.py`
  - Return scored annotation evidence/proposals instead of only boolean atom selection.
- Modify: `src/agfc/raster_content_region.py`
  - Emit weak/rejected proposals with reasons instead of silently returning `None` for recoverable cases.
- Modify: `src/agfc/figure_instance_evidence.py`
  - Convert pre-instance negative marking into arbitration features/rules, then delete the module if it becomes only a pass-through.
- Modify: `src/agfc/figure_objects.py`
  - Remove object-layer final authority such as redundant complete/content marking once arbitration consumes all hypotheses.
- Modify: `src/agfc/pipeline.py`
  - Split closure scoring from suppression; keep suppression only where physically invalid or explicitly diagnostic.
- Modify: `src/agfc/runner.py`
  - Move structural-miss/raster-split authority into normal proposal generation.
- Test: `tests/test_pdf_agfc_boundary.py`
- Test: `tests/test_pdf_agfc_figure_instances.py`
- Test: existing `tests/test_pdf_agfc_raster_content_region.py`, `tests/test_pdf_agfc_pipeline.py`, `tests/test_pdf_agfc_runner.py`

## Task 1: Baseline And Fixture Lock

- [ ] **Step 1: Add a v44 hidden-fail fixture helper**

Create test data helpers that encode the 13 hidden cases as mechanism fixtures, not page-specific golden boxes.

- [ ] **Step 2: Add failing tests for raster group scope vs boundary**

Cases:
- `jm_0004` shape: visual group bbox broader than content proposal; final bbox must not include group edges unless needed.
- `jm_0010` shape: visual group exists but right-side annotation/support proposal must still be considered.

Run: `PYTHONPATH=src pytest tests/test_pdf_agfc_figure_instances.py -k 'raster_group' -q`

- [ ] **Step 3: Add failing tests for complete object vs boundary authority**

Cases:
- padded single raster can be replaced by compact raster pixel proposal.
- coarse vector union can be demoted when one atom sets multiple outer edges.

Run: `PYTHONPATH=src pytest tests/test_pdf_agfc_figure_instances.py -k 'complete_object' -q`

- [ ] **Step 4: Add failing tests for annotation side budget**

Cases:
- missing bottom/right side can be recovered by side-specific annotation.
- hundreds of text atoms cannot be accepted as one figure extent.
- sibling-panel text cannot expand a neighboring figure's bbox.

Run: `PYTHONPATH=src pytest tests/test_pdf_agfc_figure_instances.py -k 'annotation' -q`

- [ ] **Step 5: Add failing test for same-scope sibling composition**

Case:
- two overlapping left/right proposals with shared figure/caption scope should compose into one instance or be arbitrated as one proposal set.

Run: `PYTHONPATH=src pytest tests/test_pdf_agfc_figure_instances.py -k 'sibling' -q`

## Task 2: Boundary Proposal Layer

- [ ] **Step 1: Add proposal structures to `src/agfc/boundary.py`**

Define:

```python
@dataclass(frozen=True)
class BoundaryProposal:
    id: str
    source: str
    scope_key: str | None
    support_bbox: BBox
    content_bbox: BBox
    member_atom_ids: tuple[str, ...] = ()
    excluded_atom_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    features: Mapping[str, object] = field(default_factory=dict)
    veto_features: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class BoundaryDecision:
    selected_proposal_id: str | None
    rejected_proposal_ids: tuple[str, ...]
    final_bbox: BBox
    final_member_atom_ids: tuple[str, ...]
    reasons: tuple[str, ...]
```

- [ ] **Step 2: Implement side metrics**

Add helpers for side deltas, area ratio, coverage proxy, compactness proxy, annotation density, and coarse-vector-edge suspicion.

- [ ] **Step 3: Implement proposal ranking**

Start conservative:
- prefer proposals with strong visual/member coverage;
- penalize excessive side expansion;
- penalize high annotation count/density;
- allow compact content to replace complete only when it does not drop side coverage below configurable thresholds;
- never require raw visual group bbox as a floor.

Run: `PYTHONPATH=src pytest tests/test_pdf_agfc_boundary.py -q`

## Task 3: Wire Proposals Into FigureInstance

- [ ] **Step 1: Build proposal pool in `figure_instances.py`**

Collect proposals from:
- complete object content/support bbox;
- compact candidate scores in boundary metadata;
- content branch bbox;
- primitive/raster pixel bbox;
- raster visual group bbox as scope-only proposal;
- annotation side proposals.

- [ ] **Step 2: Replace `_resolve_raster_group_boundary()`**

Delete the mandatory `bboxes = [visual_group_bbox]` behavior. Use visual group only as proposal/scope evidence.

- [ ] **Step 3: Replace `_resolve_nonraster_boundary()`**

Route complete/content/primitive choices through `BoundaryArbitrator`. Remove `_content_evidence_can_replace_complete()` and `_content_candidate_has_replacement_authority()` after parity tests pass.

- [ ] **Step 4: Allow annotation evidence with raster groups**

Remove the `visual_group_bbox is None` condition around annotation expansion. Annotation must enter as side proposals with role/density constraints.

Run: `PYTHONPATH=src pytest tests/test_pdf_agfc_figure_instances.py -q`

## Task 4: Downstream Cleanup

- [ ] **Step 1: Convert `boundary.py` to proposal generation**

Keep `calibrate_boundary()` returning the old result for compatibility, but internally expose candidate proposals. Remove local `max()` authority once all callers use proposals.

- [ ] **Step 2: Convert raster content region recoverable rejects**

For blockers/high content ratio, emit weak proposals with `rejected_reason` instead of `None` when the input is otherwise valid.

- [ ] **Step 3: Convert annotation extent to scored evidence**

Return per-atom role, side, gap, overlap, confidence, and exclusion reason. Keep boolean adapter only for legacy tests during transition.

- [ ] **Step 4: Delete object/pre-instance authority**

Move or delete:
- `figure_instance_evidence.annotate_preinstance_negative_evidence()`
- `figure_objects._annotate_redundant_object_evidence()`
- object-layer complete/content replacement marking

Keep physical-invalid hard filters only: empty bbox, zero area, missing page/image data, duplicate atoms, and true template/background rejection.

- [ ] **Step 5: Move runner two-pass raster split into proposals**

Represent structural-miss/raster-split as proposal sources in the same pool rather than a second pipeline pass.

Run: `PYTHONPATH=src pytest tests -q`

## Task 5: Diagnostics And Benchmark Gate

- [ ] **Step 1: Write proposal diagnostics**

Add per figure:
- `selected_boundary_proposal_id`
- `rejected_boundary_proposal_ids`
- `boundary_decision_reasons`
- proposal list with source, bbox, confidence, side metrics, veto features.

- [ ] **Step 2: Write candidate lifecycle diagnostics**

At minimum:
`stage`, `candidate_id`, `candidate_kind`, `bbox`, `source_atom_ids`, `decision`, `reason`, `score_components`, `suppressed_by`, `selected`.

- [ ] **Step 3: Run targeted unit tests**

Run:
`PYTHONPATH=src pytest tests/test_pdf_agfc_boundary.py tests/test_pdf_agfc_figure_instances.py -q`

- [ ] **Step 4: Run full unit suite**

Run:
`PYTHONPATH=src pytest tests -q`

- [ ] **Step 5: Run JournalMix v1 fresh benchmark**

Run the repo's existing JournalMix benchmark command for the current project convention. Compare against v44:
- reported F1 must stay at or above `0.9519` unless explicitly accepted;
- strict quality F1 must exceed `0.8269`;
- hidden quality fail matches must be below `13`;
- hard-negative FP must remain `0`.

## Rollback / Risk Notes

- Main regression risk: legitimate whitespace/gutters inside raster figures may be cropped if compact proposals are over-favored.
- Main over-expansion risk: enabling annotation evidence on raster groups can reintroduce table/body/caption text unless side and role filters are strict.
- Main architecture risk: moving closure suppression too late can increase candidate count; keep scored pruning but preserve rejected reasons and allow final arbitration to recover alternatives.
