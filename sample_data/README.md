# Sample Data

This folder contains a tiny deterministic cohort for smoke-testing the main Python pipeline bundles.

Helpers:
- `generate_sample_data.py` rebuilds the sample archives on demand.
- `run_sample_pipeline.py` runs the main pipeline scripts end to end into a scratch output directory and auto-generates the archives if they are missing.

Template configs:
- `pipeline_config.sample.json`
- `heatmap_config.sample.yaml`
