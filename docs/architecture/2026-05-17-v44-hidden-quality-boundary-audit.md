# v44 Hidden Quality Boundary Audit

Date: 2026-05-17

Scope: JournalMix v1 benchmark output at `artifacts/benchmarks/journalmix_v1/agfc_figure_instance_v44_compact_guard`.

## Executive Summary

v44's reported detector metrics are strong, but strict boundary quality exposes 13 hidden bad matches:

- Reported: precision `0.9706`, recall `0.9340`, F1 `0.9519`, mean IoU `0.9493`.
- Strict quality: precision `0.8431`, recall `0.8113`, F1 `0.8269`.
- Hidden quality-fail matches: `13`.

These are not 13 unrelated bugs. They collapse into one architectural problem: AGFC still lets multiple evidence layers make final boundary decisions before a single final arbitrator exists. Raster visual groups, complete object evidence, content branches, annotation extent, primitive/pixel refinement, closure ranking, and runner two-pass raster split all still contain local authority.

The unified fix is to separate instance scope from exported geometry. A visual group, caption scope, content branch, primitive region, annotation extent, and complete object should become boundary proposals. A single `BoundaryArbitrator`, initially implemented inside the existing `boundary.py` compatibility surface, should select or compose the exported bbox with side-aware coverage, compactness, annotation budget, and completeness checks.

## The 13 Concrete Failures

| # | Page / Prediction | Final strategy | Symptom | Root cause class |
|---:|---|---|---|---|
| 1 | `jm_0004 / panel_1` | `raster_visual_group_extent` | IoU `0.7457`; over right `+44.5pt`; purity `0.7457` | Raster group bbox is used as mandatory boundary floor |
| 2 | `jm_0010 / panel_4` | `raster_visual_group_extent` | coverage `0.8595`; crop right `-34.6pt` | Raster group path disables later semantic/annotation recovery |
| 3 | `jm_0011 / panel_3` | `content_evidence_refined_complete` | coverage `0.8961`; crop top `+20.3pt` | Primitive content branch replaces complete boundary without side coverage guard |
| 4 | `jm_0011 / panel_1` | `complete_object_evidence` | over right `+26.6pt`; purity `0.9038` | Annotation expansion crosses sibling ownership and eats right-panel text |
| 5 | `jm_0016 / panel_2` | `complete_object_evidence` | purity `0.8459`; padded raster bbox over all sides | Raster atom/branch proves identity and geometry at once |
| 6 | `jm_0036 / panel_2` | `complete_object_evidence` | purity `0.8067`; over top/bottom about `17pt` each | `single_raster_core` gets high confidence while useful pixel-content branch is demoted |
| 7 | `jm_0037 / panel_2` | `complete_object_evidence` | coverage `0.8607`; crop top `+11.1pt` | Visual union is trusted as complete geometry; annotation cannot recover missing visual side |
| 8 | `jm_0038 / panel_2` | `complete_object_evidence` | purity `0.8270`; over left/top/right | Coarse vector cluster sets outer edges and visual union gets high confidence |
| 9 | `jm_0038 / panel_3` | `complete_object_evidence` | purity `0.7978`; over left/top/right | Same coarse vector/visual-union authority issue |
| 10 | `jm_0039 / panel_1` | `complete_object_evidence` | IoU `0.7466`; crop bottom `-21.7pt` | Annotation extent is binary/late and fails side-aware recovery |
| 11 | `jm_0040 / panel_2` | `complete_object_evidence` | IoU `0.7311`; crop right `-130.2pt` | One full figure is split into overlapping sibling instances; no same-scope composition |
| 12 | `jm_0043 / panel_3` | `complete_object_evidence` | purity `0.8050`; over bottom `+15.6pt` | Coarse visual union plus broad annotation acceptance |
| 13 | `jm_0044 / panel_1` | `complete_object_evidence` | IoU `0.5842`; over top `-74.9pt`, right `+71.6pt`; 339 annotations included | Complete-object visual union and annotation expansion lack global density/role caps |

## Root Cause Clusters

### 1. Raster scope is conflated with boundary geometry

Evidence:

- `jm_0004` fully covers GT but over-expands from `[333.3, 50.5, 541.3, 251]` to `[315.2782, 44.7144, 585.8139, 251.4357]`.
- `jm_0010` crops the right side from GT `[50.8, 58, 298.3, 159]` to prediction `[50.9837, 55.3969, 263.7195, 159.1859]`.

Code:

- `build_figure_instances()` builds raster groups and assigns active objects before final boundary resolution: `src/agfc/figure_instances.py:38-43`.
- `_resolve_instance_boundary()` appends `visual_group_bbox` and routes to `_resolve_raster_group_boundary()` whenever a visual group exists: `src/agfc/figure_instances.py:267-275`.
- `_resolve_raster_group_boundary()` starts with `bboxes = [visual_group_bbox]`, then unions content branches: `src/agfc/figure_instances.py:355-393`.
- Annotation expansion is skipped when `visual_group_bbox` exists: `src/agfc/figure_instances.py:279-289`.

Conclusion: raster group should be scope evidence only. It must not be a mandatory exported bbox floor.

### 2. Complete object evidence proves identity and boundary at the same time

Evidence:

- `jm_0016`, `jm_0036`, `jm_0038` x2, `jm_0043`, and `jm_0044` are high-coverage / low-purity matches.
- `jm_0036` has a useful raster content branch with ratio about `0.6229`, but the final bbox keeps padded `single_raster_core`.
- `jm_0044` exports `[72.6022, -4.4098, 595.5626, 229.7447]` for GT `[71, 70.5, 524, 230.5]`; top/right are driven by coarse vector atoms and 339 annotation atoms are accepted.

Code:

- `boundary.py` generates support/single-raster/visual-union candidates and immediately chooses `max(scored_candidates)`: `src/agfc/boundary.py:18-45`.
- `calibrate_boundary_from_object()` can let primitive evidence override locally before instance arbitration: `src/agfc/boundary.py:126-150`.
- `_resolve_nonraster_boundary()` picks the best complete candidate by confidence/object score/area and returns its content bbox: `src/agfc/figure_instances.py:455-484`.

Conclusion: complete objects should establish "this belongs to the instance"; exported bbox must be selected from proposals with independent purity/coverage checks.

### 3. Content branch replacement is not side-aware

Evidence:

- `jm_0011 / panel_3` uses `content_evidence_refined_complete`; prediction top is `75.2584` while GT top is `55`, producing low coverage.
- The content branch is `primitive_evidence_region` with `content_to_support_area_ratio = 1.0`, so it looks authoritative even though it misses the top side.

Code:

- `_content_evidence_can_replace_complete()` mostly checks area ratio and overlap coverage, then allows replacement: `src/agfc/figure_instances.py:497-518`.
- `_content_candidate_has_replacement_authority()` grants authority based on source and ratio, not side-by-side missing extent: `src/agfc/figure_instances.py:521-532`.

Conclusion: content proposals need side metrics. A replacement cannot be accepted only because aggregate area/overlap passes.

### 4. Annotation extent is binary, late, and too global

Evidence:

- `jm_0037` and `jm_0039` remain under-covered after annotation handling.
- `jm_0011 / panel_1`, `jm_0043`, and `jm_0044` over-expand because annotation acceptance is too permissive once inside global budget.
- In `jm_0011 / panel_1`, right-side text atoms from the sibling panel push the bbox rightward; this is an ownership/scope failure, not just a width threshold issue.
- `jm_0044` accepts 339 annotation atoms, which is a strong signal that text role/density constraints are missing.

Code:

- `collect_nonraster_annotation_atoms()` filters with a boolean classifier: `src/agfc/annotation_extent.py:12-34`.
- `_select_annotation_atoms_within_budget()` greedily unions atoms if global area/width/height budgets allow it: `src/agfc/figure_instances.py:332-346`.
- The budget is not side-aware and has no annotation-count/density cap: `src/agfc/figure_instances.py:314-329`.

Conclusion: annotation extent should produce scored side proposals with role, distance, direction, and density features. It should not mutate the selected bbox by greedy union.

### 5. Sibling fragments with shared semantic scope are not composed

Evidence:

- `jm_0040 / panel_2` matches the full GT but crops right by `130.2pt`; the same page also has a right-side overlapping prediction (`panel_3`) that becomes a quality failure/FP. The two fragments should have been treated as competing/composable proposals for one full figure.

Code:

- `rank_closure_results()` still suppresses or keeps closures before object/instance arbitration: `src/agfc/pipeline.py:109-190`.
- `build_figure_instances()` groups primarily through raster visual groups and object-evidence components, not a stable semantic scope model: `src/agfc/figure_instances.py:45-69`.

Conclusion: caption/figure scope and layout sibling relations need to feed instance composition before final bbox export.

## Architecture Hotspots

- `src/agfc/figure_instances.py:36-43`: pre-instance negative evidence and active filtering still prune before final arbitration.
- `src/agfc/figure_instance_evidence.py:11-30`: table, fragment replacement, annotation replacement, and overbroad primitive decisions are still pre-instance mutation rules.
- `src/agfc/figure_objects.py:282-325`: object layer still marks redundant/complete hypotheses as negative.
- `src/agfc/boundary.py:18-45`: proposal generation and selection are coupled.
- `src/agfc/raster_content_region.py:45-150`: proposal function can return `None` for blockers/high ratio instead of emitting weak/rejected proposals.
- `src/agfc/annotation_extent.py:20-34`: text extent is a binary predicate, not scored evidence.
- `src/agfc/pipeline.py:109-190`: closure ranking still performs suppression before full instance arbitration.
- `src/agfc/runner.py:214-249`: structural miss triggers a second pass with raster split proposals outside the main candidate pool.

## Unified Fix

Add one final boundary arbitration layer.

Data model:

```python
BoundaryProposal(
    id,
    source,
    scope_key,
    support_bbox,
    content_bbox,
    member_atom_ids,
    excluded_atom_ids,
    confidence,
    features,
    veto_features,
)

BoundaryDecision(
    selected_proposal_id,
    rejected_proposal_ids,
    final_bbox,
    final_member_atom_ids,
    reasons,
)
```

Rules:

- Raster visual groups connect evidence but do not force bbox inclusion.
- Complete objects provide completeness and ownership evidence, not automatic geometry authority.
- Content branches, primitive regions, raster pixel regions, visual unions, support bboxes, annotation extents, and raster-split proposals become comparable proposals.
- Annotation proposals are side-specific and budgeted by side, count, role, and density.
- Same-scope sibling proposals may be composed before final selection.
- All rejected/weak proposals remain diagnostic evidence with reasons.

## Acceptance Criteria

- Unit tests cover the 4 mechanism clusters above with synthetic cases.
- Benchmark guard: v44 strict quality F1 improves from `0.8269`; hidden quality fails drop below `13`.
- No regression in reported F1 below `0.9519`.
- No hard-negative false positives are introduced.
- Every exported figure records `selected_boundary_proposal_id`, rejected proposal ids, and side-aware decision reasons.
