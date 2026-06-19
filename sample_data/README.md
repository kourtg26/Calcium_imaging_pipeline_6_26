# Sample Data

This folder contains a small real-data subset for smoke-testing the main Python pipeline bundles.

Current sample cohort:
- Animals: `1`, `2`, `4`, `6`
- Ext1 sample zip is copied from the real Ext1 z-scored NPZ outputs in the repo.
- Ext2 sample zip is copied from the real `ext2_raw_trace_files.zip` archive in the repo.
- Retrieval sample z-scored zip is copied from the real retrieval z-scored NPZ bundle in the repo.
- Retrieval sample raw zip is rebuilt from the real retrieval z-scored traces because the original retrieval raw CSV zip is not present in this workspace.

Classifier alignment:
- Ext1 and Ext2 sample runs use the current `3s/2-of-3` tone-onset rule (`min_hits_per_period = 2`) rather than the older `1-of-3` sample setting.
- Retrieval still uses its session-specific early/late tone definition (`Tone 1` vs `Tone 4`) from the main pipeline config.

Helpers:
- `generate_sample_data.py` rebuilds the sample archives on demand from the real source files already present in the workspace.
- `run_sample_pipeline.py` runs the main pipeline scripts end to end into a scratch output directory and auto-generates the archives if they are missing.

Template configs:
- `pipeline_config.sample.json`
- `heatmap_config.sample.yaml`
