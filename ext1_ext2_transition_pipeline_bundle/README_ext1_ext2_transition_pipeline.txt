Ext1↔Ext2 transition pipeline

Created: 2026-01-29T19:40:52Z

What you upload next
--------------------
Upload your Ext2 raw trace zips (same structure as Ext1 raw traces):
  - ext2_raw_trace_files_part1.zip
  - ext2_raw_trace_files_part2.zip
  (etc.)

Each zip should contain per-animal CSVs with:
  - Time column (Time_s preferred)
  - ToneFlag (0/1) or another binary tone flag column
  - Cell columns named consistently across days (e.g., C001, C002, ...), *and identical between Ext1 and Ext2 for registered cells*

Then run (or ask me to run here):
python /mnt/data/ext1_ext2_transition_pipeline.py \
  --ext1_zip /mnt/data/ext1_classes_proportions_only.zip \
  --ext2_raw_zips /mnt/data/ext2_raw_trace_files_part1.zip /mnt/data/ext2_raw_trace_files_part2.zip \
  --out_dir /mnt/data \
  --work_dir /mnt/data/_ext1_ext2_transition_work

Outputs (small)
---------------
- ext2_classes_proportions_only.zip
- Transition CSVs:
  - cell-weighted and animal-weighted (overall + by sex)
  - per-animal transition counts/proportions
  - EarlyOnly/LateOnly source-conditional transition tables
  - EarlyOnly/LateOnly retention-by-animal tables
  - tone-responsive retention summaries + sex-difference stats (with FDR columns where applicable)
- Alluvial plots as PNG + SVG + PDF:
  - overall + female + male
  - EarlyOnly and LateOnly source-conditional (All/Female/Male; cell-weighted and animal-weighted %)
- Sankey HTML files for EarlyOnly/LateOnly source-conditional transitions
- ext1_to_ext2_transitions_alluvial_bundle.zip (everything)

Notes
-----
If your Ext2 files use different cell_id strings than Ext1 (e.g., C001 becomes Cell_1 or has day-specific suffixes),
we can add a mapping step — but the current script assumes exact cell_id matches.
The pipeline now scans extracted Ext2 zips recursively for CSV files, so nested zip folder structures are supported.
