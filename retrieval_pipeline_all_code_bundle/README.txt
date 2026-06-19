Retrieval pipeline code package (chat-generated)

This folder contains runnable Python scripts used to:
  1) Process Retrieval raw traces:
     - z-score each cell across the entire session
     - classify onset-evoked responsiveness on Tone 1 (early) vs Tone 4 (late)
     - export per-animal per-cell class tables and per-animal proportions

  2) Compute Retrieval tone-by-tone plots:
     - for EarlyOnly and LateOnly groups, compute mean z activity during each of the 4 tones
     - overlay connected line plots (mean±SEM across animals) on freezing bar plots per tone
     - output All, Male, Female figures + CSV tables

  3) Build Ext1 → Ext2 → Retrieval transition alluvials:
     - merge per-cell classes across sessions by cell_id within animal
     - normalize cell IDs (e.g., 'undecidedC12' -> 'C012') and class labels
     - output long cell table, count tables, and alluvial plots (static + interactive)

Files:
  - pipeline_config.json: parameters & expected filenames
  - 00_retrieval_process_raw_traces.py
  - 01_retrieval_tone_by_tone_activity_freeze.py
  - 02_ext1_ext2_ret_transitions_alluvial.py
  - requirements_min.txt

Usage (from /mnt/data):
  python 00_retrieval_process_raw_traces.py --config pipeline_config.json
  python 01_retrieval_tone_by_tone_activity_freeze.py --config pipeline_config.json
  python 02_ext1_ext2_ret_transitions_alluvial.py --config pipeline_config.json

Notes:
  - The Ext2 classes zip is expected to be the UPDATED10489 version (prefix normalized).
  - Transitions are computed only for cells present in all 3 sessions (inner join).
