# AGFC JournalMix Dataset Design

**Status:** Draft for review

**Owner:** Codex + user

**Scope assumption:** This design targets a **private journal-paper figure evaluation set** only. It does **not** mix engineering reports or industry documents into the main benchmark. If a migration/control subset is needed later, it should be created as a separate follow-up set rather than folded into `v1`.

## 1. Background

AGFC is a non-neural figure recovery pipeline that operates on PDF structure and reconstructs logical figures through atom graphs, seed evidence, and closure. The current public benchmark track is already sufficient to demonstrate that AGFC can run on an open dataset and be compared with external baselines. However, the public track is still not enough to isolate AGFC's intended advantages on journal-paper pages, especially for:

- roadmap / flow / route-style figures
- compound multi-panel figures
- vector-dominant figures
- mixed vector+raster figures

The goal of this dataset is therefore **evaluation and mechanism analysis**, not training.

## 2. Goals

This dataset should support three things at once:

1. A small but credible private benchmark for journal-paper PDF pages.
2. A hard-case slice that exposes AGFC's advantages beyond standard detection-style public metrics.
3. A stable internal testbed for error review, mechanism analysis, and future paper figures/tables.

## 3. Non-Goals

This work does not aim to:

- build a training dataset
- replace the public DocLayNet benchmark
- cover all document domains
- optimize AGFC mainline algorithm code in this thread
- chase benchmark scale over benchmark sharpness

## 4. Why A Private Mixed Journal Set Is Needed

The current local public benchmark setup already contains:

- a mixed `DocLayNet` 64-page pilot
- a `single_picture_64` control slice
- MinerU comparison outputs on the same `single_picture_64` slice

Those public slices are useful for comparability, but their labels are still mostly detection-style `Picture` boxes. They are weak at evaluating:

- logical grouping completeness
- whether AGFC incorrectly fragments a single logical figure
- whether nearby body text is incorrectly absorbed
- whether a caption/description strip is retained when it should be part of the figure

The private set should therefore complement, not replace, the public track:

- `DocLayNet`: external comparability
- `JournalMix`: AGFC-specific stress testing and mechanism analysis

## 5. Dataset Positioning

The recommended dataset name is:

`AGFC-JournalMix-v1`

It should be described as:

> A small, curated, private evaluation set of born-digital journal-paper PDF pages, built to stress logical figure recovery rather than generic picture detection.

## 6. Design Principles

The dataset should follow these principles:

1. **Small but hard**
   Prefer 80-100 carefully selected pages over hundreds of low-value pages.

2. **Mechanism-oriented**
   Each page should help explain AGFC behavior, not merely increase sample count.

3. **PDF-native**
   Only include born-digital PDFs with recoverable structure. Avoid scanned pages in `v1`.

4. **Page-level benchmark**
   The annotation unit remains the page, with page-local figure GT.

5. **Journal-first**
   Only journal papers belong in the main set. Other domains can become separate control sets later.

6. **Human-verified**
   Automatic preselection and prelabeling are allowed, but final GT must be human-confirmed.

7. **Version-frozen**
   Once `v1` is frozen, later additions must go to `v1.1` or `v2`, not silently mutate `v1`.

## 7. Recommended Scale

Recommended first release:

- `84` pages total
- from `20-24` journal PDFs
- at most `4` selected pages from any one PDF

Recommended composition:

- `72` positive pages
- `12` hard-negative pages

Rationale:

- `72` positives are enough to show pattern-level differences without making adjudication too heavy.
- `12` hard negatives protect against overclaiming and help catch false-positive tendencies.
- limiting pages per paper reduces template/style leakage.

## 8. Figure-Type Mixture

The positive pages should be distributed across four primary buckets, each with `18` pages:

1. `route_map_or_flow`
   Route maps, process flows, pipeline diagrams, topology-like paths, direction-heavy figures.

2. `compound_multi_panel`
   Multi-panel scientific figures, grouped subfigures, panel grids, borderless grouped visual blocks.

3. `vector_dominant`
   Figures primarily built from PDF vector primitives, chart lines, shapes, legends, and text.

4. `mixed_vector_raster`
   Figures combining raster images with vector overlays, legends, labels, callouts, and side strips.

Each bucket should contain three difficulty layers:

- `control`: `4` pages
- `hard`: `8` pages
- `stress`: `6` pages

Difficulty guidance:

- `control`: clear target figure, low ambiguity
- `hard`: caption tight to figure, borderless, nearby text, or cross-column interaction
- `stress`: multi-figure adjacency, strong contamination risk, vector-heavy weak seeds, layout traps

## 9. Hard-Negative Slice

The `12` hard-negative pages should contain page structures that visually tempt AGFC but should not be exported as target figures in the benchmark target sense, for example:

- large tables with strong borders or shading
- pseudocode blocks with boxes/arrows
- formula-heavy blocks with diagram-like spacing
- decorative vector/icon regions
- page ornaments or sidebars

These pages should still be journal-paper pages. The point is to control false positives within the same document domain.

## 10. Source Selection Rules

### 10.1 PDF inclusion rules

Include only PDFs that satisfy all or most of the following:

- born-digital journal PDF
- stable page geometry
- extractable text blocks and vector/raster structure
- no dominant scan artifacts
- legally safe for internal evaluation use

### 10.2 Page inclusion rules

Prefer pages that satisfy at least one high-value condition:

- vector structure dominates the figure body
- multiple subpanels form one logical figure
- caption/description strip should be kept with the figure
- figure is adjacent to dense body text
- figure spans columns or interacts with column layout
- local context makes overmerge/fragmentation plausible

Avoid low-value pages such as:

- trivial single raster image with wide whitespace
- pages with only decorative icons
- pages whose figure is so tiny that even human GT is unstable

### 10.3 Diversity rules

Try to balance across:

- English and Chinese papers
- single-column and double-column layouts
- full-width and embedded figures
- short captions and long captions
- one-figure pages and multi-figure pages

## 11. Ground Truth Design

The benchmark should reuse the existing page-level JSON shape already defined in:

- `data/bench/gt_schema.json`

Each GT page should retain:

- `page_idx`
- `page_label`
- `figures`

Each figure should retain:

- `figure_id`
- `bbox`
- `logical_group_id`
- `panel_bboxes`
- `caption_bbox`
- `body_exclusion_bboxes`

### 11.1 Semantics of the fields

`bbox`

- The final logical figure region expected from AGFC export.
- It is not merely the tightest visual core if the benchmark policy says the caption/strip belongs to the figure.

`logical_group_id`

- Stable identifier for one logical figure on one page.
- Used for fragmentation/overmerge-oriented evaluation logic.

`panel_bboxes`

- The major visible subpanels that constitute the logical figure.
- Use these for structure-aware evaluation, not for over-fragmenting minor decorations.

`caption_bbox`

- Use when the caption or description strip should be retained as part of the target figure.
- Leave `null` when caption is absent or intentionally excluded from the figure target.

`body_exclusion_bboxes`

- Explicit nearby body-text regions that must not be absorbed into the exported figure.
- Use these conservatively and only for evaluation-relevant contamination boundaries.

## 12. Recommended Sidecar Metadata

To keep the core GT schema stable, store richer analysis labels in sidecar metadata rather than bloating the minimal GT.

Recommended page-level metadata fields:

- `doc_id`
- `source_pdf_name`
- `source_hash`
- `page_no`
- `language`
- `layout_mode`
- `figure_family`
- `pdf_modality`
- `difficulty`
- `has_caption_target`
- `figure_count_on_page`
- `selection_reason`
- `review_status`
- `review_notes`

Suggested values:

- `layout_mode`: `single_column`, `double_column`, `full_width`, `mixed`
- `figure_family`: `route_map_or_flow`, `compound_multi_panel`, `vector_dominant`, `mixed_vector_raster`, `hard_negative`
- `pdf_modality`: `vector`, `raster`, `mixed`
- `difficulty`: `control`, `hard`, `stress`
- `review_status`: `preselected`, `prelabeled`, `confirmed`, `adjudicated`, `frozen`

## 13. Directory Layout

The recommended on-repo dataset skeleton is:

```text
data/private/journalmix_v1/
├── README.md
├── manifest.json
├── page_index.csv
├── gt/
│   ├── jm_0001.json
│   ├── jm_0002.json
│   └── ...
├── meta/
│   ├── jm_0001.json
│   ├── jm_0002.json
│   └── ...
├── review/
│   ├── shortlist.csv
│   ├── adjudication_queue.csv
│   └── freeze_log.md
└── source_map.local.example.json
```

Notes:

- raw source PDFs should **not** be committed if they are private
- the real local mapping file should be `source_map.local.json` and stay untracked
- `manifest.json` should define dataset version, split policy, counts, and freeze date

## 14. Benchmark Construction Workflow

The build should proceed in clearly separated stages.

### Stage A: Build source pool

Goal:

- assemble `25-30` candidate journal PDFs

Output:

- `source_pool.csv` or equivalent list with provenance, language, and legal/internal-use notes

### Stage B: Automatic page mining

Goal:

- run AGFC corpus/gallery tooling on the source pool
- produce candidate pages, overlays, extracted figures, and debug bundles

Output:

- a broad candidate pool, likely `120-180` pages

### Stage C: Shortlist construction

Goal:

- reduce candidate pages to around `120` pages
- ensure bucket balance before annotation starts

Rules:

- do not let one paper dominate
- do not take too many trivial pages just because they were easy to mine

### Stage D: Prelabeling

Goal:

- create annotation scaffolds and draft GT

This can be assisted by:

- AGFC overlay bundles
- predicted figure boxes
- panel candidates
- seed and closure debug outputs

But these are only aids. They must not become unreviewed GT.

### Stage E: Human confirmation

Goal:

- confirm or correct each candidate page's GT

This is where the benchmark becomes valid.

### Stage F: Adjudication

Goal:

- resolve ambiguous pages
- freeze difficult caption/body boundary decisions consistently

### Stage G: Freeze

Goal:

- cut `AGFC-JournalMix-v1`
- stop silent mutation

Outputs:

- frozen `manifest.json`
- final page count and bucket counts
- freeze note describing what was included and excluded

## 15. Collaboration Mode

The working mode for this dataset should be:

> **Codex leads implementation; user provides bounded confirmations.**

This is the default operating policy for `AGFC-JournalMix-v1`.

### 15.1 What Codex should lead

Codex should own:

- source-pool structuring
- running corpus and mining candidate pages
- generating overlays and review bundles
- assembling shortlist and bucket balance reports
- creating annotation scaffolds
- drafting GT from visible evidence
- schema validation
- dataset packaging
- manifest writing
- freeze bookkeeping

### 15.2 What the user should confirm

The user should provide lightweight but high-value confirmation on:

- which private journal PDFs are allowed into the source pool
- pages with real semantic ambiguity
- pages whose caption inclusion is debatable
- pages whose figure boundaries depend on domain meaning rather than geometry alone
- final freeze approval

### 15.3 What should not require user interruption

Codex may proceed without interruption on:

- candidate mining
- page deduplication
- quota balancing drafts
- generating scaffold files
- obvious low-value page rejection
- obvious schema-normalization edits
- routine metadata completion

### 15.4 What must trigger user confirmation

Codex should pause for confirmation when:

- a source PDF's permission/eligibility is unclear
- a page is high-value but boundary semantics are ambiguous
- a figure may or may not include a caption/strip with real paper-facing consequences
- a page could fit multiple buckets and the choice affects benchmark balance
- the frozen set composition changes materially

## 16. Human Effort Budget

The design should minimize user involvement while preserving benchmark credibility.

Target operating model:

- Codex handles `80-90%` of mechanical work
- user handles `10-20%` of high-judgment review

In practical terms, the user should mainly see:

- source-pool approval
- a compact ambiguity queue
- final freeze review

The user should **not** need to manually inspect every candidate page.

## 17. Review Policy

To keep the set trustworthy:

- all `stress` pages must be explicitly reviewed by a human
- at least `20%` of all positive pages should receive secondary review
- any page with disputed caption inclusion should be logged in adjudication notes
- hard negatives should be reviewed with the same care as positives

If review bandwidth is tight, prefer:

- fewer pages
- higher confidence

over:

- more pages
- weakly reviewed GT

## 18. Versioning And Freeze Policy

Freeze policy for `v1`:

- no silent edits after freeze
- any GT correction after freeze must be logged
- if corrections are material, release `v1.1`

Recommended manifest fields:

- `dataset_name`
- `version`
- `frozen_at`
- `page_count`
- `positive_page_count`
- `hard_negative_page_count`
- `doc_count`
- `bucket_counts`
- `review_policy`
- `notes`

## 19. Success Criteria

`AGFC-JournalMix-v1` should be considered successful if it achieves all of the following:

1. It is small enough to finish in a focused build cycle.
2. It is hard enough to expose AGFC-specific strengths and failure modes.
3. It is structured enough to support reproducible evaluation.
4. It does not require the user to hand-label everything.
5. It is frozen clearly enough to support paper writing and future comparisons.

## 20. Recommended Immediate Execution Order

The next steps after approving this design should be:

1. create the dataset skeleton under `data/private/journalmix_v1/`
2. define the source-pool manifest format
3. collect the first batch of journal PDFs allowed for internal use
4. run automatic candidate mining
5. generate a first shortlist
6. review only the ambiguity queue and freeze policy before full annotation starts

## 21. Open Decisions Deferred To A Later Step

The following decisions are intentionally deferred and should not block the design:

- whether to create a separate migration/control subset later
- whether evaluator-side contamination/overmerge logic should be expanded before or after `v1` freeze
- whether to publish a redacted or metadata-only public companion package

## 22. Final Recommendation

Proceed with `AGFC-JournalMix-v1` as a **private, journal-only, mixed hard-case benchmark** with:

- `84` pages total
- `72` positives + `12` hard negatives
- four positive buckets with balanced difficulty layers
- page-level JSON GT using the existing schema
- sidecar metadata for mechanism analysis
- a collaboration mode where Codex performs the heavy lifting and the user confirms only bounded, high-value decisions

This gives AGFC a benchmark that is:

- small enough to finish
- sharp enough to matter
- controlled enough to trust
- aligned with the current repository architecture and evaluation direction
