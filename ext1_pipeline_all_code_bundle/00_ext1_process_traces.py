#!/usr/bin/env python3
"""
00_ext1_process_traces.py

Inputs:
  - Ext1 z-scored NPZ zips (paths.ext1_zscored_npz_parts)

Outputs:
  - ext1_classes_proportions_only.zip
      * Ext1_classes_proportions_perAnimal.csv
      * Ext1_cellClassifications_long.csv
      * {animal}_ext1_onset_evoked_cell_classes.csv (per animal)

Classification rule:
  - uses tone_onset_classification_ext1 if present; otherwise tone_onset_classification_ext2
  - onset-evoked if delta >= thr_z for >= consec_frames within 0..window_s post onset
  - early/late tones defined by config lists
"""
import os, json, argparse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from freezing_pipeline_all_code_bundle.utils_freezing import (
    extract_npz_parts, load_ext1_npz, classify_onset_evoked, standardize_class_label
)

CLASSES = ["EarlyOnly", "Overlap", "LateOnly", "Neither"]


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)


def _norm_animal_id(a: str) -> str:
    s = str(a).strip()
    if s.lower().startswith("animal"):
        s = s.lower().replace("animal", "", 1)
    return s

def sex_of(animal_id, female_set, male_set):
    a = _norm_animal_id(animal_id)
    if a in female_set:
        return "Female"
    if a in male_set:
        return "Male"
    return "Unknown"


def normalize_cell_id_any(cid: str, width: int = 3) -> str:
    s = str(cid).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return "C" + digits.zfill(width)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pipeline_config.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["paths"]["data_dir"]
    out_dir = cfg["paths"].get("output_dir", data_dir)

    female_set = set(cfg.get("cohort_sex_map", {}).get("female", []))
    male_set = set(cfg.get("cohort_sex_map", {}).get("male", []))

    parts = cfg["paths"]["ext1_zscored_npz_parts"]
    ext1_npz = extract_npz_parts(parts, out_dir)

    outputs = cfg.get("outputs", {})
    class_dir = os.path.join(out_dir, outputs.get("ext1_classes_dir", "ext1_classes"))
    Path(class_dir).mkdir(parents=True, exist_ok=True)

    summary_csv = os.path.join(out_dir, outputs.get("ext1_summary_csv", "Ext1_classes_proportions_perAnimal.csv"))
    long_csv = os.path.join(out_dir, outputs.get("ext1_cell_long_csv", "Ext1_cellClassifications_long.csv"))

    p = cfg.get("tone_onset_classification_ext1", cfg.get("tone_onset_classification_ext2", {}))

    per_rows = []
    long_rows = []
    written_class_files = []

    for npz_path in ext1_npz:
        d = load_ext1_npz(npz_path)
        aid = os.path.basename(npz_path).replace("_ext1_zscored_traces.npz", "")

        tone_cls = classify_onset_evoked(
            d["z"], d["tone_flag"], d["tone_id"], d["time"],
            thr_z=p.get("thr_z", 0.5),
            consec_frames=p.get("consec_frames", 10),
            window_s=p.get("window_s", 3.0),
            early_tones=p.get("early_tones", [1, 2, 3]),
            late_tones=p.get("late_tones", [10, 11, 12]),
            min_hits_per_period=p.get("min_hits_per_period", 2),
        )

        cell_ids = [normalize_cell_id_any(cid, 3) for cid in d["cell_ids"]]
        cls_df = pd.DataFrame({"cell_id": cell_ids, "class": tone_cls})
        cls_df["class"] = cls_df["class"].map(standardize_class_label)
        cls_path = os.path.join(class_dir, f"{aid}_ext1_onset_evoked_cell_classes.csv")
        cls_df.to_csv(cls_path, index=False)
        written_class_files.append(cls_path)

        counts = cls_df["class"].value_counts().reindex(CLASSES, fill_value=0)
        N = int(counts.sum())
        per_rows.append({
            "animal_id": aid,
            "sex": sex_of(aid, female_set, male_set),
            "n_cells": N,
            **{f"n_{k}": int(v) for k, v in counts.items()},
            **{f"p_{k}": float(v / N) if N > 0 else np.nan for k, v in counts.items()},
        })

        tmp = cls_df.copy()
        tmp["animal_id"] = aid
        tmp["sex"] = sex_of(aid, female_set, male_set)
        long_rows.append(tmp)

    per_anim = pd.DataFrame(per_rows).sort_values("animal_id")
    long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    per_anim.to_csv(summary_csv, index=False)
    long.to_csv(long_csv, index=False)

    out_zip = os.path.join(out_dir, "ext1_classes_proportions_only.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_csv, arcname=os.path.basename(summary_csv))
        zf.write(long_csv, arcname=os.path.basename(long_csv))
        for fp in sorted(set(written_class_files)):
            zf.write(fp, arcname=os.path.basename(fp))

    print("Wrote:", out_zip)


if __name__ == "__main__":
    main()
