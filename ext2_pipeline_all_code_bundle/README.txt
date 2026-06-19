Ext2 pipeline code package (chat-generated)

This folder contains runnable Python scripts used to:
  1) Process Ext2 raw traces:
     - classify onset-evoked responsiveness using tone-indexed onsets
     - export per-animal per-cell class tables and per-animal proportions

  2) Compute Ext2 tone-by-tone plots:
     - for EarlyOnly and LateOnly groups, compute mean z activity during selected tones
     - overlay connected line plots (mean±SEM across animals) on freezing bar plots per tone

Files:
  - 00_ext2_process_raw_traces.py
  - 01_ext2_tone_by_tone_activity_freeze.py

Usage (from project root):
  python ext2_pipeline_all_code_bundle/00_ext2_process_raw_traces.py --config pipeline_config.json
  python ext2_pipeline_all_code_bundle/01_ext2_tone_by_tone_activity_freeze.py --config pipeline_config.json

Notes:
  - Tone selection for tone-by-tone plots can be controlled via the "tone_by_tone.Ext2" config section.
