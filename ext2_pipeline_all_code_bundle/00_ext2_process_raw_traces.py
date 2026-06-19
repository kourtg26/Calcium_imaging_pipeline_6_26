#!/usr/bin/env python3
"""
00_ext2_process_raw_traces.py

Inputs:
  - Ext2 raw trace ZIP(s) (paths.ext2_raw_zip or paths.ext2_raw_zips)

Outputs:
  - ext2_classes_proportions_only.zip
      * Ext2_classes_proportions_perAnimal.csv
      * Ext2_cellClassifications_long.csv
      * {animal}_ext2_onset_evoked_cell_classes.csv (per animal)
"""
import os, json, argparse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from freezing_pipeline_all_code_bundle.utils_freezing import (
    extract_csvs_from_zip, choose_one_ext2_csv_per_animal, load_ext2_session_csv,
    classify_onset_evoked, standardize_class_label
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

    female_set = set(cfg.get("cohort_sex_map", {}).get("female", []))
    male_set = set(cfg.get("cohort_sex_map", {}).get("male", []))

    raw_zips = cfg["paths"].get("ext2_raw_zips", cfg["paths"].get("ext2_raw_zip"))
    if isinstance(raw_zips, str):
        raw_zips = [raw_zips]
    if not raw_zips:
        raise ValueError("paths.ext2_raw_zip(s) required")

    tmp_dir = os.path.join(data_dir, "_tmp_ext2_raw")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    csvs = []
    for zp in raw_zips:
        csvs.extend(extract_csvs_from_zip(zp, tmp_dir))
    ext2_files = choose_one_ext2_csv_per_animal(csvs)

    outputs = cfg.get("outputs", {})
    class_dir = os.path.join(data_dir, outputs.get("ext2_classes_dir", "ext2_classes"))
    Path(class_dir).mkdir(parents=True, exist_ok=True)

    summary_csv = os.path.join(data_dir, outputs.get("ext2_summary_csv", "Ext2_classes_proportions_perAnimal.csv"))
    long_csv = os.path.join(data_dir, outputs.get("ext2_cell_long_csv", "Ext2_cellClassifications_long.csv"))

    p = cfg.get("tone_onset_classification_ext2", {})

    per_rows = []
    long_rows = []
    written_class_files = []

    for fp in ext2_files:
        d = load_ext2_session_csv(fp)
        if d is None:
            continue
        aid = os.path.basename(fp).split("_")[0]
        if str(aid).lower() == "cell":
            aid = "10492"

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
        cls_path = os.path.join(class_dir, f"{aid}_ext2_onset_evoked_cell_classes.csv")
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

    out_zip = os.path.join(data_dir, "ext2_classes_proportions_only.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(summary_csv, arcname=os.path.basename(summary_csv))
        zf.write(long_csv, arcname=os.path.basename(long_csv))
        for fp in sorted(set(written_class_files)):
            zf.write(fp, arcname=os.path.basename(fp))

    print("Wrote:", out_zip)


if __name__ == "__main__":
    main()
