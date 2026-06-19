# Calcium Imaging Pipeline

Public snapshot of a calcium imaging analysis codebase for Ext1, Ext2, and Retrieval sessions.

This release is organized around runnable Python pipeline bundles, a deterministic sample dataset generator, and a smoke test that exercises the main analysis paths without requiring private lab data.

## What This Public Repo Includes
- `ext1_pipeline_all_code_bundle/` for Ext1 trace processing and tone-by-tone summaries
- `ext2_pipeline_all_code_bundle/` for Ext2 raw trace processing and onset-evoked classification
- `retrieval_pipeline_all_code_bundle/` for retrieval processing and Ext1->Ext2->Retrieval transition summaries
- `ext1_ext2_transition_pipeline_bundle/` for Ext1->Ext2 transition analysis
- `ext2_ret_transition_pipeline_bundle/` for Ext2->Retrieval transition analysis
- `freezing_pipeline_all_code_bundle/` for freezing-aligned analyses
- `heatmap_pipeline_cli_config_bundle/` for configurable tone-only heatmaps
- `standalone_plot_workflows_bundle/` for post-release plotting and statistics scripts
- `sample_data/` for a deterministic toy dataset and an end-to-end runner

## What This Public Repo Does Not Include
- private experimental source data
- machine-specific startup paths
- internal GitHub upload chunk bundles

## Requirements
- Python 3.10+
- Core packages: `numpy`, `pandas`, `matplotlib`
- Common extras: `scipy`, `plotly`, `pyyaml`

Install the public release dependencies with:

```bash
pip install -r requirements-public.txt
```

## Quick Start
Use the sample dataset first. It is the fastest way to confirm the public release runs correctly on your machine.

## Run With Sample Data
This repo now includes a small real-data sample cohort under `sample_data/` plus a Python runner that exercises the main pipeline bundles end to end.

Rebuild the sample archives manually if you want to refresh them:
```bash
python sample_data/generate_sample_data.py
```

Run the verified sample pipeline into a scratch output directory:
```bash
python sample_data/run_sample_pipeline.py --output-dir /tmp/grin-sample-run
```

If the sample archives are missing, that runner will generate them automatically before starting.

For sample runs, Ext1 and Ext2 use the current `3 s / 2-of-3` onset classifier so the demo outputs stay aligned with the newer production-style session classification.

That command runs:
- Ext1 classification and tone-block summaries
- Ext2 classification and tone-block summaries
- Retrieval classification and tone-by-tone summaries
- Ext1→Ext2→Retrieval alluvial outputs
- Ext1→Ext2 transition pipeline
- Ext2→Retrieval transition pipeline
- Freezing analyses bundle
- Heatmap generation

Run the smoke test:
```bash
python -m unittest tests.test_sample_pipeline
```

## Sample Workflow
The sample pipeline runs these stages end to end:
- Ext1 classification and tone-block summaries
- Ext2 classification and tone-block summaries
- Retrieval classification and tone-by-tone summaries
- Ext1->Ext2->Retrieval alluvial outputs
- Ext1->Ext2 transition analysis
- Ext2->Retrieval transition analysis
- Freezing analyses
- Heatmap generation

Helpful sample files:
- `sample_data/generate_sample_data.py`
- `sample_data/pipeline_config.sample.json`
- `sample_data/heatmap_config.sample.yaml`

## Main Entry Points
Canonical entrypoints live in the bundle directories rather than the repository root.

Ext1:
- `ext1_pipeline_all_code_bundle/00_ext1_process_traces.py`
- `ext1_pipeline_all_code_bundle/01_ext1_tone_by_tone_activity_freeze.py`

Ext2:
- `ext2_pipeline_all_code_bundle/00_ext2_process_raw_traces.py`
- `ext2_pipeline_all_code_bundle/01_ext2_tone_by_tone_activity_freeze.py`

Retrieval:
- `retrieval_pipeline_all_code_bundle/00_retrieval_process_raw_traces.py`
- `retrieval_pipeline_all_code_bundle/01_retrieval_tone_by_tone_activity_freeze.py`
- `retrieval_pipeline_all_code_bundle/02_ext1_ext2_ret_transitions_alluvial.py`

Transitions:
- `ext1_ext2_transition_pipeline_bundle/ext1_ext2_transition_pipeline.py`
- `ext2_ret_transition_pipeline_bundle/ext2_ret_transition_pipeline.py`

Freezing:
- `freezing_pipeline_all_code_bundle/run_all_freezing_analyses.py`

Heatmaps:
- `heatmap_pipeline_cli_config_bundle/make_heatmap.py`

## Config
- `pipeline_config.example.json` is the safest starting template for adapting the pipeline to a new dataset.
- `pipeline_config.json` is a fuller merged example covering Ext1, Ext2, Retrieval, freezing, and transitions.

## Notes
- Keep `animal_id` and `cell_id` naming consistent across sessions if you need cross-session registration.
- Some workflows support recursive CSV discovery after zip extraction.
- Some root-level scripts are preserved as historical wrappers or legacy entrypoints, but the bundle directories above are the preferred paths for new use.
