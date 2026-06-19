# Freezing analyses code bundle (Ext1 / Ext2 / Retrieval)

This folder contains the Python scripts used to generate the **freezing-related plots/analyses** in this chat:
- Freezing-onset Δz mean traces (animal-weighted & cell-weighted), tone-only freezing epochs
- Same traces restricted to tone-onset responsive cells
- Split-by-tone-onset-class freezing-onset traces
- Freeze-onset Δz summarized by early/late tone subsets
- Freeze-onset vs tone-onset activity correlations
- Overlap of "freezing-responsive" cells with tone-onset responsive classes

## Inputs expected (same naming used in chat)
Put these files in the same directory as the scripts, or edit paths in `config_template.json`:
- `ext1_zscored_npz_part1.zip`, `ext1_zscored_npz_part2.zip`, `ext1_zscored_npz_part3.zip`
- `ext1_classes_proportions_only.zip`
- `ext2_raw_trace_files.zip`
- `ret_zscored_npz_bundle.zip`
- `ret_classes_proportions_only.zip`

## Environment
Python 3.9+ with:
- numpy
- pandas
- matplotlib
- scipy (for correlations)

## How to run
Each script is runnable as a standalone module:
```bash
python 01_freezeOnset_deltaZ_meanTraces_allSessions.py --config config_template.json
python 02_freezeOnset_deltaZ_meanTraces_responsiveOnly_allSessions.py --config config_template.json
python 03_freezeOnset_deltaZ_meanTraces_splitByClass_allSessions.py --config config_template.json
python 04_freezeOnset_vs_toneOnset_allSessions.py --config config_template.json
python 05_freezeOnset_deltaZ_meanByToneSubset_splitByClass.py --config config_template.json
python 06_freezeOnsetDeltaZ_meanTraces_byToneSubset_splitByClass_ANIMALWEIGHTED.py --config config_template.json
python 07_freezeOnsetDeltaZ_meanTraces_byToneSubset_splitByClass_CELLWEIGHTED.py --config config_template.json
python 08_freezeResponsive_overlapToneOnset_byClass.py --config config_template.json
```

Outputs are written into `output_dir` (default `.`).

## Notes
- **Tone-only freezing onsets** = rising edges of FreezeFlag where ToneFlag==1.
- **Freezing-onset Δz** (default): mean(z[0..1s]) − mean(z[−1..0]).
- **Tone-onset peakMean 0..3s** (default): per tone onset, max(z in 0..3s), then averaged across tones for each cell.

Generated: 2026-02-03
