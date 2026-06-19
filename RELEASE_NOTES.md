# Release Notes

## 2026-06-19

### Public Release Cleanup
- Reframed the repository as a public-facing runnable code snapshot with a clearer top-level README.
- Removed internal GitHub upload chunk bundles from the public release.
- Switched the sample-data workflow to regenerate archives on demand instead of relying on checked-in generated artifacts.
- Kept the sample smoke test as the main public validation path.

## 2026-04-06

### Repository Cleanup
- Removed exact duplicate root-level files where bundle copies already existed, keeping the bundle paths as the canonical entrypoints.
- Removed legacy Ext1→Ext2 carry-over files from `ext2_ret_transition_pipeline_bundle/` so that bundle only contains Ext2→Retrieval assets.
- Renamed the old root helper scripts `Heatmap` and `Mean_trace_plots` to `legacy_ext1_heatmap.py` and `legacy_ext1_mean_trace_plots.py` for clarity.

### Standalone Plot Workflow Bundle
- Added `standalone_plot_workflows_bundle/` to package standalone plotting/statistics workflows created since the `2026-03-05` release sync.
- Bundled post-release standalone scripts into `standalone_plot_workflows_bundle/scripts/`, including:
  - session-context activity plots (`make_session_first120s_toneResponsive_comparisons.py`)
  - classified session-anchor mean traces (`make_classified_session_anchor_mean_traces.py`)
  - cross-session matching/correlation scripts
  - AUC and retention/statistics plotting scripts
  - animal 5 representative-image montage scripts
- Added `scripts_manifest.csv` to classify the bundled standalone scripts by workflow type.

### Representative Output Bundles
- Added copied example outputs to `standalone_plot_workflows_bundle/outputs/` for:
  - `session_first120s_vs_firstToneSet_toneResponsive_bundle`
  - `classified_sessionAnchor_meanTrace_bundle`
  - `representative_cell_images_animal5`

### Release Sync
- Updated top-level and release READMEs to document the standalone plot-workflow bundle and example outputs.
- Synced the new standalone bundle into `grin_pipeline_release/` and `grin_pipeline_release 6/`.

## 2026-03-05

### Transition Pipeline Updates
- Updated `ext1_ext2_transition_pipeline_bundle/ext1_ext2_transition_pipeline.py` to emit expanded transition outputs:
  - per-animal transition counts/proportions
  - EarlyOnly/LateOnly source-conditional transition tables
  - EarlyOnly/LateOnly retention-by-animal tables
  - source retention summaries and sex-difference stats (Welch + permutation p-values with BH-FDR columns)
  - source-conditional alluvial and Sankey outputs (All/Female/Male; cell-weighted and animal-weighted %)
- Updated `ext2_ret_transition_pipeline_bundle/ext2_ret_transition_pipeline.py` with the same expanded transition outputs plus retrieval-denominator retention summaries and sex-comparison CSVs.
- Added thin black outlines to alluvial flow bands and optional percent labels for source-conditional plots.
- Added recursive CSV discovery after zip extraction so nested zip folder layouts are supported.

### Metadata/Bundling
- Transition run metadata now includes ID-alignment diagnostics and expanded output coverage.
- Transition bundle zips now include the full expanded `Ext1toExt2_*` and `Ext2toRet_*` output sets.

### Documentation
- Updated release README to reflect new transition outputs and usage scope.
