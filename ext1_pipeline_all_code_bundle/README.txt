Ext1 pipeline code package (chat-generated)

This folder contains runnable Python scripts used to:
  1) Process Ext1 z-scored traces:
     - classify onset-evoked responsiveness using tone-indexed onsets
     - export per-animal per-cell class tables and per-animal proportions

  2) Compute Ext1 tone-by-tone plots:
     - for EarlyOnly and LateOnly groups, compute mean z activity during selected tones
     - overlay connected line plots (mean±SEM across animals) on freezing bar plots per tone

Files:
  - 00_ext1_process_traces.py
  - 01_ext1_tone_by_tone_activity_freeze.py

Usage (from project root):
  python ext1_pipeline_all_code_bundle/00_ext1_process_traces.py --config pipeline_config.json
  python ext1_pipeline_all_code_bundle/01_ext1_tone_by_tone_activity_freeze.py --config pipeline_config.json

Notes:
  - Tone selection for tone-by-tone plots can be controlled via the "tone_by_tone.Ext1" config section.
