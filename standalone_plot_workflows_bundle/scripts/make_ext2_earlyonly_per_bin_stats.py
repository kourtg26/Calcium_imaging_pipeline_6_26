#!/usr/bin/env python3
"""
Per-bin Early vs Late tone-onset statistics for Ext2 onset-defined cells.

Defaults target the updated 2/3, 0.5z, 3 s post-onset criteria:
  - class source: ext2_pipeline_output_clean_3s2of3/ext2_classes
  - cells: Ext2 EarlyOnly by default
  - compared events: tones 1-3 vs tones 10-12
  - trace: delta from onset, -5 to +5 s, 100 ms bins

Primary inferential test:
  - paired two-sided Wilcoxon signed-rank test per bin across cells
  - Benjamini-Hochberg FDR correction across bins

Supplementary output:
  - paired t-test per bin, also FDR-corrected
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from freezing_pipeline_all_code_bundle.utils_freezing import (
    choose_one_ext2_csv_per_animal,
    extract_csvs_from_zip,
    infer_fps,
    load_ext2_session_csv,
    standardize_class_label,
)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def norm_animal_id(value: str) -> str:
    s = str(value).strip()
    if s.lower() == "cell":
        return "10492"
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    if s.lower().startswith("animal"):
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            return f"animal{int(digits)}"
    return s


def normalize_cell_id_any(cell_id: str, width: int = 3) -> str:
    s = str(cell_id).strip()
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
        # Binary in-tone vectors should be treated like tone flags, not a
        # one-epoch tone-id sequence.
        if len(uniq) > 1:
            for tone_num in sorted(uniq):
                idxs = np.where(tid == tone_num)[0]
                if len(idxs) == 0:
                    continue
                epochs.append({"tone_num": tone_num, "onset_idx": int(idxs[0]), "idxs": idxs})
            epochs = sorted(epochs, key=lambda d: d["onset_idx"])
            for order, epoch in enumerate(epochs, start=1):
                epoch["tone_order"] = order
            return epochs

    flag = pd.to_numeric(np.asarray(tone_flag), errors="coerce")
    flag = np.where(np.isfinite(flag), flag, 0.0)
    flag = (flag > 0).astype(int)
    rises = np.where((flag[1:] == 1) & (flag[:-1] == 0))[0] + 1
    falls = np.where((flag[1:] == 0) & (flag[:-1] == 1))[0] + 1
    if flag[0] == 1:
        rises = np.r_[0, rises]
    if flag[-1] == 1:
        falls = np.r_[falls, n]
    for order, (start, stop) in enumerate(zip(rises, falls), start=1):
        epochs.append(
            {
                "tone_num": order,
                "tone_order": order,
                "onset_idx": int(start),
                "idxs": np.arange(start, stop),
            }
        )
    return epochs


def extract_cell_psth(z: np.ndarray, onsets: list[int], cell_idx: int, pre_f: int, post_f: int):
    traces = []
    for onset in onsets:
        if onset - pre_f < 0 or onset + post_f >= z.shape[0]:
            continue
        seg = z[onset - pre_f : onset + post_f + 1, cell_idx].astype(float)
        traces.append(seg - z[onset, cell_idx])
    if not traces:
        return None, 0
    arr = np.vstack(traces)
    return np.nanmean(arr, axis=0), int(arr.shape[0])


def cohen_dz(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)


def paired_wilcoxon(first: np.ndarray, last: np.ndarray):
    mask = np.isfinite(first) & np.isfinite(last)
    x = first[mask]
    y = last[mask]
    if x.size == 0:
        return np.nan, np.nan, 0
    diff = x - y
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 0.0, 1.0, int(x.size)
    stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(stat), float(p), int(x.size)


def paired_ttest(first: np.ndarray, last: np.ndarray):
    mask = np.isfinite(first) & np.isfinite(last)
    x = first[mask]
    y = last[mask]
    if x.size < 2:
        return np.nan, np.nan, int(x.size)
    res = stats.ttest_rel(x, y, nan_policy="omit")
    return float(res.statistic), float(res.pvalue), int(x.size)


def load_class_map(class_dir: Path) -> dict[str, dict[str, str]]:
    class_map: dict[str, dict[str, str]] = {}
    for fp in sorted(class_dir.glob("*_ext2_onset_evoked_cell_classes.csv")):
        animal_prefix = fp.name.split("_ext2_", 1)[0].strip()
        if animal_prefix.lower() == "cell":
            continue
        if not re.fullmatch(r"(animal\d+|\d{4,5})", animal_prefix, flags=re.IGNORECASE):
            continue
        animal = norm_animal_id(animal_prefix)
        df = pd.read_csv(fp)
        if "cell_id" not in df.columns or "class" not in df.columns:
            continue
        df["cell_id"] = df["cell_id"].astype(str).map(lambda x: normalize_cell_id_any(x, 3))
        df["class"] = df["class"].map(standardize_class_label)
        class_map[animal] = dict(zip(df["cell_id"], df["class"]))
    return class_map


def build_cell_trace_matrices(
    root: Path,
    config: dict,
    class_dir: Path,
    pre_s: float,
    post_s: float,
    selected_class: str,
):
    raw_zips = config["paths"].get("ext2_raw_zips", config["paths"].get("ext2_raw_zip"))
    if isinstance(raw_zips, str):
        raw_zips = [raw_zips]

    tmp_dir = root / "_tmp_ext2_raw_per_bin_stats"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csvs = []
    for zip_path in raw_zips:
        csvs.extend(extract_csvs_from_zip(zip_path, str(tmp_dir)))
    ext2_files = choose_one_ext2_csv_per_animal(csvs)

    class_map = load_class_map(class_dir)
    common_t = np.arange(-pre_s, post_s + 1e-9, 0.1)
    cell_rows = []

    for fp in ext2_files:
        session = load_ext2_session_csv(fp)
        if session is None:
            continue

        animal_raw = os.path.basename(fp).split("_")[0]
        animal_key = norm_animal_id(animal_raw)
        cmap = class_map.get(animal_key, {})
        if not cmap:
            continue

        epochs = sorted(
            find_tone_epochs_from_arrays(session["tone_id"], session["tone_flag"]),
            key=lambda e: e["onset_idx"],
        )
        if len(epochs) < 12:
            continue

        first_onsets = [e["onset_idx"] for e in epochs[:3]]
        last_onsets = [e["onset_idx"] for e in epochs[-3:]]

        z = np.asarray(session["z"], dtype=float)
        cell_ids = [normalize_cell_id_any(c, 3) for c in session["cell_ids"]]
        fps = infer_fps(session["time"])
        if not np.isfinite(fps):
            fps = 10.0
        pre_f = int(round(pre_s * fps))
        post_f = int(round(post_s * fps))
        native_t = np.arange(-pre_f, post_f + 1) / fps

        for idx, cell_id in enumerate(cell_ids):
            if cmap.get(cell_id) != selected_class:
                continue
            first_trace, n_first = extract_cell_psth(z, first_onsets, idx, pre_f, post_f)
            last_trace, n_last = extract_cell_psth(z, last_onsets, idx, pre_f, post_f)
            if first_trace is None or last_trace is None:
                continue
            cell_rows.append(
                {
                    "animal_id": animal_key,
                    "cell_id": cell_id,
                    "source_file": os.path.basename(fp),
                    "fps": float(fps),
                    "n_first_events_used": int(n_first),
                    "n_last_events_used": int(n_last),
                    "first3_trace": np.interp(common_t, native_t, first_trace),
                    "last3_trace": np.interp(common_t, native_t, last_trace),
                }
            )

    if not cell_rows:
        raise RuntimeError("No usable cells found for the requested class and trace window.")

    first_mat = np.vstack([row["first3_trace"] for row in cell_rows])
    last_mat = np.vstack([row["last3_trace"] for row in cell_rows])
    return common_t, first_mat, last_mat, pd.DataFrame(cell_rows)


def make_stats_table(common_t: np.ndarray, first_mat: np.ndarray, last_mat: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, time_s in enumerate(common_t):
        first = first_mat[:, idx]
        last = last_mat[:, idx]
        diff = first - last
        diff_mean = float(np.nanmean(diff))
        t_stat, p_t, n_t = paired_ttest(first, last)
        w_stat, p_w, n_w = paired_wilcoxon(first, last)
        rows.append(
            {
                "time_s": float(time_s),
                "n_cells": int(np.sum(np.isfinite(first) & np.isfinite(last))),
                "mean_first3": float(np.nanmean(first)),
                "sem_first3": float(np.nanstd(first, ddof=1) / math.sqrt(n_t)) if n_t > 1 else np.nan,
                "mean_last3": float(np.nanmean(last)),
                "sem_last3": float(np.nanstd(last, ddof=1) / math.sqrt(n_t)) if n_t > 1 else np.nan,
                "mean_diff_first_minus_last": diff_mean,
                "cohen_dz": cohen_dz(diff),
                "ttest_rel_t": t_stat,
                "ttest_rel_p": p_t,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": p_w,
            }
        )

    stats_df = pd.DataFrame(rows)

    for p_col, q_col, sig_col in [
        ("ttest_rel_p", "ttest_rel_p_fdr_bh", "ttest_rel_sig_fdr_bh_0p05"),
        ("wilcoxon_p", "wilcoxon_p_fdr_bh", "wilcoxon_sig_fdr_bh_0p05"),
    ]:
        pvals = stats_df[p_col].to_numpy(dtype=float)
        valid = np.isfinite(pvals)
        qvals = np.full_like(pvals, np.nan)
        sig = np.zeros_like(pvals, dtype=bool)
        if valid.any():
            reject, qvals_valid, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
            qvals[valid] = qvals_valid
            sig[valid] = reject
        stats_df[q_col] = qvals
        stats_df[sig_col] = sig

    return stats_df


def plot_with_significance(common_t: np.ndarray, stats_df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    first_mean = stats_df["mean_first3"].to_numpy()
    first_sem = stats_df["sem_first3"].to_numpy()
    last_mean = stats_df["mean_last3"].to_numpy()
    last_sem = stats_df["sem_last3"].to_numpy()

    ax.plot(common_t, first_mean, linewidth=2.4, label="Tones 1-3", color="#1f77b4")
    ax.fill_between(common_t, first_mean - first_sem, first_mean + first_sem, color="#1f77b4", alpha=0.2)
    ax.plot(common_t, last_mean, linewidth=2.4, label="Tones 10-12", color="#ff7f0e")
    ax.fill_between(common_t, last_mean - last_sem, last_mean + last_sem, color="#ff7f0e", alpha=0.2)

    ax.axvline(0, color="steelblue", alpha=0.5)
    ax.axhline(0, color="steelblue", alpha=0.2)
    ax.set_xlabel("Time from tone onset (s)")
    ax.set_ylabel("Delta z from onset")

    sig_mask = stats_df["wilcoxon_sig_fdr_bh_0p05"].to_numpy(dtype=bool)
    y_min = np.nanmin(np.r_[first_mean - first_sem, last_mean - last_sem])
    y_max = np.nanmax(np.r_[first_mean + first_sem, last_mean + last_sem])
    y_range = y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
    marker_y = y_min - 0.08 * y_range
    if sig_mask.any():
        ax.scatter(common_t[sig_mask], np.full(sig_mask.sum(), marker_y), s=18, marker="s", color="black", label="Wilcoxon FDR < 0.05")

    ax.set_ylim(marker_y - 0.05 * y_range, y_max + 0.08 * y_range)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def make_prefix(selected_class: str) -> str:
    return f"Ext2_{selected_class}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Project root containing data and outputs.",
    )
    ap.add_argument(
        "--config",
        default="grin_pipeline_release 4/pipeline_config_ext2.json",
        help="Path to the ext2 pipeline config, relative to --root unless absolute.",
    )
    ap.add_argument(
        "--class-dir",
        default="ext2_pipeline_output_clean_3s2of3/ext2_classes",
        help="Directory containing per-animal Ext2 class CSVs, relative to --root unless absolute.",
    )
    ap.add_argument(
        "--output-dir",
        default="ext2_psth_earlyOnly_allCells_first3_vs_last3_updatedCriteria_2of3_3s",
        help="Directory for stats outputs, relative to --root unless absolute.",
    )
    ap.add_argument("--selected-class", default="EarlyOnly")
    ap.add_argument("--pre-s", type=float, default=5.0)
    ap.add_argument("--post-s", type=float, default=5.0)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    class_dir = Path(args.class_dir)
    if not class_dir.is_absolute():
        class_dir = root / class_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    prefix = make_prefix(args.selected_class)
    common_t, first_mat, last_mat, cell_df = build_cell_trace_matrices(
        root=root,
        config=config,
        class_dir=class_dir,
        pre_s=args.pre_s,
        post_s=args.post_s,
        selected_class=args.selected_class,
    )

    stats_df = make_stats_table(common_t, first_mat, last_mat)

    cell_out = output_dir / f"{prefix}_cells_used_first3_vs_last3_updatedCriteria_2of3_3s.csv"
    stats_out = output_dir / f"{prefix}_per_bin_early_vs_late_stats_updatedCriteria_2of3_3s.csv"
    plot_out = output_dir / f"{prefix}_per_bin_early_vs_late_stats_updatedCriteria_2of3_3s.png"
    meta_out = output_dir / f"{prefix}_per_bin_early_vs_late_stats_metadata_updatedCriteria_2of3_3s.json"

    cell_df.drop(columns=["first3_trace", "last3_trace"]).sort_values(["animal_id", "cell_id"]).to_csv(cell_out, index=False)
    stats_df.to_csv(stats_out, index=False)

    sig_bins = int(stats_df["wilcoxon_sig_fdr_bh_0p05"].sum())
    title = (
        f"Ext2 {args.selected_class} cells: per-bin Early vs Late tests\n"
        f"Paired Wilcoxon primary, BH-FDR across bins; n={len(cell_df)} cells, sig bins={sig_bins}"
    )
    plot_with_significance(common_t, stats_df, plot_out, title)

    meta = {
        "classification_source": str(class_dir / "Ext2_cellClassifications_long.csv"),
        "selected_class": args.selected_class,
        "criteria": {
            "thr_z": 0.5,
            "consec_frames": 10,
            "window_s": 3.0,
            "min_hits_per_period": 2,
            "early_tones": [1, 2, 3],
            "late_tones": [10, 11, 12],
        },
        "trace_window_s": {"pre": args.pre_s, "post": args.post_s},
        "trace_baseline": "delta from onset frame (trace - z_at_onset)",
        "weighting": "cell-weighted paired-by-cell",
        "primary_test": "paired two-sided Wilcoxon signed-rank test",
        "secondary_test": "paired two-sided t-test",
        "multiple_comparisons": "Benjamini-Hochberg FDR across time bins",
        "n_cells_total": int(len(cell_df)),
        "n_animals_with_cells": int(cell_df["animal_id"].nunique()),
        "n_time_bins": int(len(common_t)),
        "n_significant_bins_wilcoxon_fdr_bh_0p05": sig_bins,
    }
    meta_out.write_text(json.dumps(meta, indent=2))

    print("Wrote:", stats_out)
    print("Wrote:", plot_out)
    print("Wrote:", meta_out)
    print("n_cells:", len(cell_df))
    print("sig_bins_wilcoxon_fdr_bh_0p05:", sig_bins)


if __name__ == "__main__":
    main()
