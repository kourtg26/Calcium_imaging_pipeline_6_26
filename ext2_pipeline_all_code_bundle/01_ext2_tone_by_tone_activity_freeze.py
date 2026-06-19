#!/usr/bin/env python3
"""
01_ext2_tone_by_tone_activity_freeze.py

Uses:
  - per-animal per-cell classes from 00_* (ext2_classes directory or zip)
  - Ext2 raw trace CSVs

Outputs:
  - Ext2_toneByTone_activity_freezing_perAnimal.csv
  - Ext2_toneByTone_summary_meanSEM.csv
  - Ext2_toneByTone_activityLines_freezeBars_{ALL,Male,Female}.png
  - ext2_toneByTone_activity_freeze_bundle.zip
"""
import os, glob, zipfile, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from freezing_pipeline_all_code_bundle.utils_freezing import (
    extract_csvs_from_zip, choose_one_ext2_csv_per_animal, load_ext2_session_csv, standardize_class_label, infer_fps
)


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return json.load(f)


def _norm_animal_id(a: str) -> str:
    s = str(a).strip()
    if s.lower().startswith("animal"):
        s = s.lower().replace("animal", "", 1)
    if s.lower() == "cell":
        s = "10492"
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


def find_tone_epochs_from_arrays(tone_id, tone_flag):
    n = len(tone_flag)
    epochs = []
    if tone_id is not None and np.any(np.asarray(tone_id) > 0):
        tid = pd.to_numeric(np.asarray(tone_id), errors="coerce")
        uniq = [int(x) for x in np.unique(tid[np.isfinite(tid)]) if int(x) != 0]
        if len(uniq) > 0:
            for u in sorted(uniq):
                idxs = np.where(tid == u)[0]
                if len(idxs) == 0:
                    continue
                epochs.append({"tone_num": u, "onset_idx": int(idxs[0]), "idxs": idxs})
            epochs = sorted(epochs, key=lambda d: d["onset_idx"])
            for k, e in enumerate(epochs, start=1):
                e["tone_order"] = k
            return epochs
    # fallback: use tone_flag transitions
    flag = pd.to_numeric(np.asarray(tone_flag), errors="coerce")
    flag = np.where(np.isfinite(flag), flag, 0.0)
    flag = (flag > 0).astype(int)
    rises = np.where((flag[1:] == 1) & (flag[:-1] == 0))[0] + 1
    falls = np.where((flag[1:] == 0) & (flag[:-1] == 1))[0] + 1
    if flag[0] == 1:
        rises = np.r_[0, rises]
    if flag[-1] == 1:
        falls = np.r_[falls, n]
    for k, (s, e) in enumerate(zip(rises, falls), start=1):
        idxs = np.arange(s, e)
        epochs.append({"tone_num": k, "tone_order": k, "onset_idx": int(s), "idxs": idxs})
    return epochs


def select_epochs(epochs, cfg_section):
    tone_orders = cfg_section.get("tone_orders") if cfg_section else None
    tone_ids = cfg_section.get("tone_ids") if cfg_section else None
    tone_count = int(cfg_section.get("tone_count", 12)) if cfg_section else 12
    if tone_orders:
        keep = [e for e in epochs if e.get("tone_order") in tone_orders]
    elif tone_ids:
        keep = [e for e in epochs if e.get("tone_num") in tone_ids]
    else:
        keep = epochs[:tone_count]
    return keep


def block_ids():
    return {
        "1-3": [1, 2, 3],
        "4-6": [4, 5, 6],
        "7-9": [7, 8, 9],
        "10-12": [10, 11, 12],
    }


def mean_sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, 0
    sem = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    return float(np.mean(x)), sem, int(len(x))


def plot_tone_by_tone(summary_df, sex_label, out_png):
    s = summary_df[summary_df["sex"] == sex_label].sort_values("block_order")
    x = np.arange(len(s))
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    freeze_mean = s["freeze_mean"].fillna(0).values
    freeze_sem = s["freeze_sem"].fillna(0).values
    ax1.bar(x, freeze_mean, yerr=freeze_sem, capsize=4)
    ax1.set_xlabel("Tone block")
    ax1.set_ylabel("Freezing (fraction during tone)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(s["block"].values)
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    early_mean = s["early_z_mean"].fillna(0).values
    early_sem = s["early_z_sem"].fillna(0).values
    late_mean = s["late_z_mean"].fillna(0).values
    late_sem = s["late_z_sem"].fillna(0).values
    ax2.errorbar(x, early_mean, yerr=early_sem, marker="o", linewidth=2, label="EarlyOnly")
    ax2.errorbar(x, late_mean, yerr=late_sem, marker="o", linewidth=2, label="LateOnly")
    ax2.set_ylabel("Mean z activity during tone")
    ax2.axhline(0, linewidth=1, alpha=0.4)
    ax2.legend(frameon=False, loc="upper right")
    ax1.set_title(f"Ext2 tone-block activity (first 3s) with freezing ({sex_label})\nAnimal-weighted mean ± SEM")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def event_aligned_delta_traces(z, onsets, preF, postF):
    kept = []
    for idx in onsets:
        if idx - preF < 0 or idx + postF >= z.shape[0]:
            continue
        kept.append(idx)
    if len(kept) == 0:
        return None
    kept = np.array(kept, dtype=int)
    traces = []
    for idx in kept:
        seg = z[idx - preF:idx + postF + 1, :]
        base = z[idx:idx + 1, :]
        traces.append(seg - base)
    return np.stack(traces, axis=0)  # n_events x T x C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pipeline_config.json")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = cfg["paths"]["data_dir"]
    out_dir = cfg["paths"].get("output_dir", data_dir)
    female_set = set(cfg.get("cohort_sex_map", {}).get("female", []))
    male_set = set(cfg.get("cohort_sex_map", {}).get("male", []))

    outputs = cfg.get("outputs", {})
    class_dir = os.path.join(out_dir, outputs.get("ext2_classes_dir", "ext2_classes"))
    Path(class_dir).mkdir(parents=True, exist_ok=True)

    ext2_zip = cfg["paths"].get("ext2_classes_zip")
    if ext2_zip and not glob.glob(os.path.join(class_dir, "*_ext2_onset_evoked_cell_classes.csv")):
        with zipfile.ZipFile(ext2_zip, "r") as zf:
            zf.extractall(class_dir)

    class_files = sorted(glob.glob(os.path.join(class_dir, "*_ext2_onset_evoked_cell_classes.csv")))
    class_map = {}
    for fp in class_files:
        base = os.path.basename(fp)
        suffix = "_ext2_onset_evoked_cell_classes.csv"
        if not base.endswith(suffix):
            continue
        animal = base[:-len(suffix)]
        animal_key = _norm_animal_id(animal)
        if animal_key == "":
            continue
        d = pd.read_csv(fp)
        d["cell_id"] = d["cell_id"].astype(str)
        d["cell_id"] = d["cell_id"].map(lambda x: normalize_cell_id_any(x, 3))
        d["class"] = d["class"].map(standardize_class_label)
        class_map[animal_key] = d.set_index("cell_id")["class"].to_dict()

    raw_zips = cfg["paths"].get("ext2_raw_zips", cfg["paths"].get("ext2_raw_zip"))
    if isinstance(raw_zips, str):
        raw_zips = [raw_zips]

    tmp_dir = os.path.join(data_dir, "_tmp_ext2_raw")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    csvs = []
    for zp in raw_zips:
        csvs.extend(extract_csvs_from_zip(zp, tmp_dir))
    ext2_files = choose_one_ext2_csv_per_animal(csvs)

    tone_cfg = (cfg.get("tone_by_tone", {}) or {}).get("Ext2", {})
    trace_cfg = (cfg.get("tone_trace", {}) or {}).get("Ext2", {})
    trace_pre_s = float(trace_cfg.get("pre_s", 5.0))
    trace_post_s = float(trace_cfg.get("post_s", 5.0))
    trace_first_n = int(trace_cfg.get("first_n", 3))
    trace_last_n = int(trace_cfg.get("last_n", 3))

    rows = []
    trace_rows = []
    trace_rows_cell_weighted = []
    for fp in ext2_files:
        d = load_ext2_session_csv(fp)
        if d is None:
            continue
        animal_raw = os.path.basename(fp).split("_")[0]
        animal_key = _norm_animal_id(animal_raw)
        cmap = class_map.get(animal_key, {})

        epochs_all = find_tone_epochs_from_arrays(d["tone_id"], d["tone_flag"])
        epochs_all = sorted(epochs_all, key=lambda e: e["onset_idx"])
        if len(epochs_all) == 0:
            continue
        epochs = select_epochs(epochs_all, tone_cfg)

        freeze = np.asarray(d["freeze"], dtype=float)
        freeze = np.where(np.isfinite(freeze), freeze, 0.0)
        freeze = (freeze > 0).astype(int)

        Z = np.asarray(d["z"], dtype=float)
        cell_ids = [normalize_cell_id_any(c, 3) for c in d["cell_ids"]]
        cls_arr = np.array([cmap.get(cid, "NA") for cid in cell_ids], dtype=object)
        early_mask = cls_arr == "EarlyOnly"
        late_mask = cls_arr == "LateOnly"

        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        winF = int(round(3.0 * fps))

        # Tone-block summaries (1-3, 4-6, 7-9, 10-12)
        blocks = block_ids()
        block_vals = {}
        for block_name, tone_orders in blocks.items():
            keep = [e for e in epochs_all if e.get("tone_order") in tone_orders]
            if len(keep) == 0:
                block_vals[block_name] = (np.nan, np.nan, np.nan)
                continue
            fvals = []
            e_vals = []
            l_vals = []
            for e in keep:
                idxs = e["idxs"]
                fvals.append(float(np.mean(freeze[idxs])) if len(idxs) > 0 else np.nan)
                onset = e["onset_idx"]
                win = Z[onset:onset + winF, :]
                if win.shape[0] == 0:
                    e_vals.append(np.nan)
                    l_vals.append(np.nan)
                else:
                    e_vals.append(float(np.nanmean(win[:, early_mask])) if early_mask.any() else np.nan)
                    l_vals.append(float(np.nanmean(win[:, late_mask])) if late_mask.any() else np.nan)
            block_vals[block_name] = (
                float(np.nanmean(fvals)) if len(fvals) else np.nan,
                float(np.nanmean(e_vals)) if len(e_vals) else np.nan,
                float(np.nanmean(l_vals)) if len(l_vals) else np.nan,
            )

        row = {
            "animal_id": animal_key,
            "sex": sex_of(animal_key, female_set, male_set),
            "n_early_cells": int(early_mask.sum()),
            "n_late_cells": int(late_mask.sum()),
        }
        for block_name, (f_mean, e_mean, l_mean) in block_vals.items():
            row[f"freeze_block_{block_name}"] = f_mean
            row[f"early_z_block_{block_name}"] = e_mean
            row[f"late_z_block_{block_name}"] = l_mean
        rows.append(row)

        # Mean traces: first N vs last N tones, delta z from onset
        epochs_sorted = epochs_all
        first_epochs = epochs_sorted[:trace_first_n]
        last_epochs = epochs_sorted[-trace_last_n:] if len(epochs_sorted) >= trace_last_n else []

        for label, group_epochs in [("first", first_epochs), ("last", last_epochs)]:
            onsets = [e["onset_idx"] for e in group_epochs]
            preF = int(round(trace_pre_s * fps))
            postF = int(round(trace_post_s * fps))
            traces = event_aligned_delta_traces(Z, onsets, preF, postF)
            if traces is None:
                continue
            t_axis = np.arange(-preF, postF + 1) / fps
            for cls_name, mask in [("EarlyOnly", early_mask), ("LateOnly", late_mask)]:
                if not np.any(mask):
                    continue
                cls_tr = traces[:, :, mask]
                ev_mean = np.nanmean(cls_tr, axis=0)
                cell_mean = np.nanmean(ev_mean, axis=1)
                trace_rows.append({
                    "animal_id": animal_key,
                    "sex": sex_of(animal_key, female_set, male_set),
                    "class": cls_name,
                    "group": label,
                    "t_axis": t_axis,
                    "trace": cell_mean,
                })
                # Cell-weighted variant: keep one trace per cell (event-averaged), pooled across animals.
                for ci in range(ev_mean.shape[1]):
                    trace_rows_cell_weighted.append({
                        "animal_id": animal_key,
                        "sex": sex_of(animal_key, female_set, male_set),
                        "class": cls_name,
                        "group": label,
                        "t_axis": t_axis,
                        "trace": ev_mean[:, ci],
                    })

    tone_df = pd.DataFrame(rows).sort_values("animal_id")
    tone_csv = os.path.join(out_dir, "Ext2_toneBlock_activity_freezing_perAnimal.csv")
    tone_df.to_csv(tone_csv, index=False)

    summary_rows = []
    block_order_list = ["1-3", "4-6", "7-9", "10-12"]
    for sex in ["All", "Male", "Female"]:
        sub = tone_df if sex == "All" else tone_df[tone_df["sex"] == sex]
        for block in block_order_list:
            fcol = f"freeze_block_{block}"
            ecol = f"early_z_block_{block}"
            lcol = f"late_z_block_{block}"
            m_f, sem_f, n_f = mean_sem(sub[fcol]) if fcol in sub.columns else (np.nan, np.nan, 0)
            m_e, sem_e, n_e = mean_sem(sub[ecol]) if ecol in sub.columns else (np.nan, np.nan, 0)
            m_l, sem_l, n_l = mean_sem(sub[lcol]) if lcol in sub.columns else (np.nan, np.nan, 0)
            summary_rows.append({
                "sex": sex,
                "block": block,
                "block_order": block_order_list.index(block),
                "freeze_mean": m_f,
                "freeze_sem": sem_f,
                "freeze_n": n_f,
                "early_z_mean": m_e,
                "early_z_sem": sem_e,
                "early_z_n": n_e,
                "late_z_mean": m_l,
                "late_z_sem": sem_l,
                "late_z_n": n_l,
            })

    summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, "Ext2_toneBlock_summary_meanSEM.csv")
    summary.to_csv(summary_csv, index=False)

    png_all = os.path.join(out_dir, "Ext2_toneBlock_activityLines_freezeBars_ALL.png")
    png_m = os.path.join(out_dir, "Ext2_toneBlock_activityLines_freezeBars_Male.png")
    png_f = os.path.join(out_dir, "Ext2_toneBlock_activityLines_freezeBars_Female.png")
    plot_tone_by_tone(summary, "All", png_all)
    plot_tone_by_tone(summary, "Male", png_m)
    plot_tone_by_tone(summary, "Female", png_f)

    # mean trace outputs (first vs last tones)
    if trace_rows:
        common_t = np.arange(-trace_pre_s, trace_post_s + 1e-9, 0.1)
        for cls_name in ["EarlyOnly", "LateOnly"]:
            for label in ["first", "last"]:
                trs = [r for r in trace_rows if r["class"] == cls_name and r["group"] == label]
                if not trs:
                    continue
                mat = np.vstack([np.interp(common_t, r["t_axis"], r["trace"]) for r in trs])
                mean = np.nanmean(mat, axis=0)
                sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.full_like(mean, np.nan)
                out_csv = os.path.join(out_dir, f"Ext2_toneTrace_{cls_name}_{label}3_meanSEM.csv")
                pd.DataFrame({"time_s": common_t, "mean": mean, "sem": sem}).to_csv(out_csv, index=False)

            first = [r for r in trace_rows if r["class"] == cls_name and r["group"] == "first"]
            last = [r for r in trace_rows if r["class"] == cls_name and r["group"] == "last"]
            if first and last:
                fmat = np.vstack([np.interp(common_t, r["t_axis"], r["trace"]) for r in first])
                lmat = np.vstack([np.interp(common_t, r["t_axis"], r["trace"]) for r in last])
                fmean = np.nanmean(fmat, axis=0)
                lmean = np.nanmean(lmat, axis=0)
                fsem = np.nanstd(fmat, axis=0, ddof=1) / np.sqrt(fmat.shape[0]) if fmat.shape[0] > 1 else np.full_like(fmean, np.nan)
                lsem = np.nanstd(lmat, axis=0, ddof=1) / np.sqrt(lmat.shape[0]) if lmat.shape[0] > 1 else np.full_like(lmean, np.nan)
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                ax.plot(common_t, fmean, label="First 3", linewidth=2)
                ax.fill_between(common_t, fmean - fsem, fmean + fsem, alpha=0.2)
                ax.plot(common_t, lmean, label="Last 3", linewidth=2)
                ax.fill_between(common_t, lmean - lsem, lmean + lsem, alpha=0.2)
                ax.axvline(0, alpha=0.6)
                ax.axhline(0, alpha=0.25)
                ax.set_xlabel("Time from tone onset (s)")
                ax.set_ylabel("Δz from onset")
                ax.set_title(f"Ext2 {cls_name}: first 3 vs last 3 tones (Δz)")
                ax.legend(frameon=False)
                plt.tight_layout()
                out_png = os.path.join(out_dir, f"Ext2_toneTrace_{cls_name}_first3_vs_last3.png")
                plt.savefig(out_png, dpi=300)
                plt.close()

    # mean trace outputs (first vs last tones) - cell-weighted
    if trace_rows_cell_weighted:
        common_t = np.arange(-trace_pre_s, trace_post_s + 1e-9, 0.1)
        for cls_name in ["EarlyOnly", "LateOnly"]:
            for label in ["first", "last"]:
                trs = [r for r in trace_rows_cell_weighted if r["class"] == cls_name and r["group"] == label]
                if not trs:
                    continue
                mat = np.vstack([np.interp(common_t, r["t_axis"], r["trace"]) for r in trs])
                mean = np.nanmean(mat, axis=0)
                sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.full_like(mean, np.nan)
                out_csv = os.path.join(out_dir, f"Ext2_toneTrace_{cls_name}_{label}3_meanSEM_cellWeighted.csv")
                pd.DataFrame({"time_s": common_t, "mean": mean, "sem": sem}).to_csv(out_csv, index=False)

            first = [r for r in trace_rows_cell_weighted if r["class"] == cls_name and r["group"] == "first"]
            last = [r for r in trace_rows_cell_weighted if r["class"] == cls_name and r["group"] == "last"]
            if first and last:
                fmat = np.vstack([np.interp(common_t, r["t_axis"], r["trace"]) for r in first])
                lmat = np.vstack([np.interp(common_t, r["t_axis"], r["trace"]) for r in last])
                fmean = np.nanmean(fmat, axis=0)
                lmean = np.nanmean(lmat, axis=0)
                fsem = np.nanstd(fmat, axis=0, ddof=1) / np.sqrt(fmat.shape[0]) if fmat.shape[0] > 1 else np.full_like(fmean, np.nan)
                lsem = np.nanstd(lmat, axis=0, ddof=1) / np.sqrt(lmat.shape[0]) if lmat.shape[0] > 1 else np.full_like(lmean, np.nan)
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                ax.plot(common_t, fmean, label="First 3", linewidth=2)
                ax.fill_between(common_t, fmean - fsem, fmean + fsem, alpha=0.2)
                ax.plot(common_t, lmean, label="Last 3", linewidth=2)
                ax.fill_between(common_t, lmean - lsem, lmean + lsem, alpha=0.2)
                ax.axvline(0, alpha=0.6)
                ax.axhline(0, alpha=0.25)
                ax.set_xlabel("Time from tone onset (s)")
                ax.set_ylabel("Δz from onset")
                ax.set_title(f"Ext2 {cls_name}: first 3 vs last 3 tones (Δz, cell-weighted)")
                ax.legend(frameon=False)
                plt.tight_layout()
                out_png = os.path.join(out_dir, f"Ext2_toneTrace_{cls_name}_first3_vs_last3_cellWeighted.png")
                plt.savefig(out_png, dpi=300)
                plt.close()

    out_zip = os.path.join(out_dir, "ext2_toneBlock_activity_freeze_bundle.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in [tone_csv, summary_csv, png_all, png_m, png_f]:
            if os.path.exists(fp):
                zf.write(fp, arcname=os.path.basename(fp))

    print("Wrote:", out_zip)


if __name__ == "__main__":
    main()
