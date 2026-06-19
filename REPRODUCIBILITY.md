# Reproducibility Guide

This document explains what can be reproduced directly from the public repository, what additional inputs are required to reproduce manuscript results, and the recommended order of operations.

## What This Repository Is

This repository is a script-based, config-driven calcium imaging analysis workflow for:

- Ext1 session processing
- Ext2 session processing
- Retrieval session processing
- cross-session transition analyses
- freezing-aligned analyses
- tone-only heatmaps
- downstream figure and statistics workflows

The code is organized by analysis stage rather than as a single installable Python package. The main bundles are:

- `ext1_pipeline_all_code_bundle/`
- `ext2_pipeline_all_code_bundle/`
- `retrieval_pipeline_all_code_bundle/`
- `ext1_ext2_transition_pipeline_bundle/`
- `ext2_ret_transition_pipeline_bundle/`
- `freezing_pipeline_all_code_bundle/`
- `heatmap_pipeline_cli_config_bundle/`
- `standalone_plot_workflows_bundle/scripts/`

## Two Levels of Reproducibility

There are two different reproducibility goals for this repository.

### 1. Reproducing The Public Code Path

This means confirming that the public release runs successfully on a new machine and produces the expected analysis outputs on the bundled sample dataset.

This is the recommended first step for all users.

### 2. Reproducing Manuscript Results

This means regenerating the class summaries, transition tables, statistics, and final manuscript figures from the original study data and the exact analysis settings used for the manuscript.

This requires private experimental inputs that are not fully included in the public release.

## Public-Code Reproduction

### Requirements

- Python 3.10 or newer
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `plotly`
- `pyyaml`

Install dependencies:

```bash
pip install -r requirements-public.txt
```

### Verify The Sample Dataset Workflow

Run the sample pipeline:

```bash
python sample_data/run_sample_pipeline.py --output-dir /tmp/grin-sample-run
```

If you want to force regeneration of the sample input archives first:

```bash
python sample_data/run_sample_pipeline.py --output-dir /tmp/grin-sample-run --regenerate-data
```

Run the smoke test:

```bash
python -m unittest tests.test_sample_pipeline
```

This verifies that the public code can:

- classify Ext1 cells
- classify Ext2 cells
- classify Retrieval cells
- compute Ext1→Ext2→Retrieval alluvial outputs
- run Ext1→Ext2 transition analyses
- run Ext2→Retrieval transition analyses
- run freezing analyses
- generate heatmaps

## Manuscript Reproduction Requirements

To reproduce manuscript results, you will need the exact manuscript-era inputs and settings.

### Required Input Files

At minimum:

- Ext1 z-scored NPZ archive(s)
- Ext2 raw trace ZIP archive
- Retrieval raw trace ZIP archive
- Retrieval z-scored NPZ ZIP archive
- any precomputed class ZIPs used for downstream plotting, if applicable
- any matched-cell or registration information required to preserve cross-session `cell_id` alignment

### Required Metadata

- exact `animal_id` naming used in the manuscript
- exact `cell_id` naming used across sessions
- cohort sex map
- session-specific tone definitions
- exact classifier thresholds used in the manuscript
- any inclusion or exclusion list for animals, cells, or sessions

### Required Analysis Provenance

- exact repository commit hash used for the manuscript
- exact config JSON used for the manuscript
- exact standalone plotting/statistics scripts used to generate each figure
- any post-processing or figure selection rules applied after the core pipeline outputs were generated

## Recommended Manuscript Reproduction Procedure

### Step 1. Freeze The Analysis Version

Record:

- the repository commit hash
- Python version
- package versions
- the final manuscript config file

### Step 2. Create A Manuscript-Specific Config

Start from:

- `pipeline_config.example.json`

Fill in:

- `paths.data_dir`
- `paths.output_dir`
- `paths.ext1_zscored_npz_parts`
- `paths.ext2_raw_zip`
- `paths.ret_raw_zip`
- `paths.ret_zscored_npz_zip`
- `cohort_sex_map`
- tone-onset classification settings
- freezing metric settings
- tone subset definitions

### Step 3. Run The Core Session Pipelines

Run in this order:

1. `ext1_pipeline_all_code_bundle/00_ext1_process_traces.py`
2. `ext1_pipeline_all_code_bundle/01_ext1_tone_by_tone_activity_freeze.py`
3. `ext2_pipeline_all_code_bundle/00_ext2_process_raw_traces.py`
4. `ext2_pipeline_all_code_bundle/01_ext2_tone_by_tone_activity_freeze.py`
5. `retrieval_pipeline_all_code_bundle/00_retrieval_process_raw_traces.py`
6. `retrieval_pipeline_all_code_bundle/01_retrieval_tone_by_tone_activity_freeze.py`
7. `retrieval_pipeline_all_code_bundle/02_ext1_ext2_ret_transitions_alluvial.py`

These steps generate the base class summaries and session-level outputs.

### Step 4. Run Cross-Session Transition Pipelines

Run:

1. `ext1_ext2_transition_pipeline_bundle/ext1_ext2_transition_pipeline.py`
2. `ext2_ret_transition_pipeline_bundle/ext2_ret_transition_pipeline.py`

These steps generate:

- matched transition tables
- cell-weighted and animal-weighted summaries
- EarlyOnly and LateOnly source-conditional outputs
- retention tables
- sex comparison outputs
- alluvial and Sankey plots

### Step 5. Run Freezing Analyses

Run:

```bash
cd freezing_pipeline_all_code_bundle
python run_all_freezing_analyses.py --config /path/to/manuscript_config.json
```

This generates freezing-aligned outputs, split-by-class summaries, and freezing-vs-tone relationships.

### Step 6. Run Heatmap Generation

Run:

```bash
python heatmap_pipeline_cli_config_bundle/make_heatmap.py --config /path/to/heatmap_config.yaml
```

### Step 7. Run Manuscript Figure Scripts

After the core outputs exist, run only the standalone scripts needed for the manuscript figures from:

- `standalone_plot_workflows_bundle/scripts/`

Examples include:

- session AUC summary workflows
- matched-cell crossed-session analyses
- tone-responsive retention proportion analyses
- session correlation workflows
- representative-image or montage scripts

Because the manuscript figure layer is modular, the exact scripts needed depend on the figure panel being reproduced.

## What To Archive For Full Reproducibility

For a manuscript reproduction package, archive all of the following:

- final config JSON file
- heatmap config YAML file
- repository commit hash
- package versions
- core output directories
- class summary CSVs
- per-cell long classification CSVs
- transition tables
- freezing summary tables
- manuscript figure scripts used
- final figures and statistics tables

## What The Public Repository Does Not Guarantee By Itself

The public repository is sufficient to reproduce the workflow structure and verify that the analysis code runs.

It is not sufficient by itself to reproduce manuscript figures exactly unless you also have:

- the original study data
- the exact manuscript config values
- the exact list of figure-generation scripts used
- any manuscript-specific inclusion, exclusion, and post-processing decisions

## Suggested Citation Language

If you are documenting reproducibility for a manuscript, a useful summary statement is:

"The public repository reproduces the analysis workflow and sample outputs. Exact reproduction of manuscript figures and statistics requires the original experimental input files, the manuscript-specific configuration file, the exact repository commit used for the analysis, and the standalone figure-generation scripts used for each panel."
