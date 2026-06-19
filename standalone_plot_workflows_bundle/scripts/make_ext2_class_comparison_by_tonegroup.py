#!/usr/bin/env python3
"""
Compare Ext2 tone-onset classes within a selected tone group.

Default use:
  - updated 2/3, 0.5z, 3 s criteria
  - compare EarlyOnly vs LateOnly
  - selected tones: first3 (tones 1-3)

Outputs:
  - class-wise PSTH mean/SEM CSV
  - per-bin independent-group stats with BH-FDR
  - window tests
  - AUC tests
  - compact summary panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from make_ext2_earlyonly_per_bin_stats import build_cell_trace_matrices, load_json


def class_prefix(class_a: str, class_b: str, tone_group: str) -> str:
    return f"Ext2_{class_a}_vs_{class_b}_{tone_group}"


def compare_independent(a: np.ndarray, b: np.ndarray):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return {
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": np.nan,
            "mean_b": np.nan,
            "mean_diff_a_minus_b": np.nan,
            "median_diff_a_minus_b": np.nan,
            "mannwhitney_u": np.nan,
            "mannwhitney_p": np.nan,
            "welch_t": np.nan,
            "welch_p": np.nan,
            "cohen_d": np.nan,
        }

    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    welch = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")

    pooled_num = ((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1))
    pooled_den = len(a) + len(b) - 2
    pooled_sd = np.sqrt(pooled_num / pooled_den) if pooled_den > 0 else np.nan
    cohen_d = (np.mean(a) - np.mean(b)) / pooled_sd if np.isfinite(pooled_sd) and pooled_sd != 0 else np.nan

    return {
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff_a_minus_b": float(np.mean(a) - np.mean(b)),
        "median_diff_a_minus_b": float(np.median(a) - np.median(b)),
        "mannwhitney_u": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
        "welch_t": float(welch.statistic),
        "welch_p": float(welch.pvalue),
        "cohen_d": float(cohen_d) if np.isfinite(cohen_d) else np.nan,
    }


def add_fdr(df: pd.DataFrame, raw_col: str, out_col: str, sig_col: str):
    pvals = df[raw_col].to_numpy(dtype=float)
    valid = np.isfinite(pvals)
    qvals = np.full_like(pvals, np.nan)
    sig = np.zeros_like(pvals, dtype=bool)
    if valid.any():
        reject, q_valid, _, _ = multipletests(pvals[valid], alpha=0.05, method="fdr_bh")
        qvals[valid] = q_valid
        sig[valid] = reject
    df[out_col] = qvals
    df[sig_col] = sig


def tone_label(tone_group: str) -> str:
    return "Tones 1-3" if tone_group == "first3" else "Tones 10-12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(REPO_ROOT),
    )
    ap.add_argument(
        "--config",
        default="grin_pipeline_release 4/pipeline_config_ext2.json",
    )
    ap.add_argument(
        "--class-dir",
        default="ext2_pipeline_output_clean_3s2of3/ext2_classes",
    )
    ap.add_argument(
        "--output-dir",
        default="ext2_compare_EarlyOnly_vs_LateOnly_tones1to3_updatedCriteria_2of3_3s",
    )
    ap.add_argument("--class-a", default="EarlyOnly")
    ap.add_argument("--class-b", default="LateOnly")
    ap.add_argument("--tone-group", choices=["first3", "last3"], default="first3")
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

    cfg = load_json(config_path)
    prefix = class_prefix(args.class_a, args.class_b, args.tone_group)

    common_t_a, first_a, last_a, cells_a = build_cell_trace_matrices(
        root=root,
        config=cfg,
        class_dir=class_dir,
        pre_s=args.pre_s,
        post_s=args.post_s,
        selected_class=args.class_a,
    )
    common_t_b, first_b, last_b, cells_b = build_cell_trace_matrices(
        root=root,
        config=cfg,
        class_dir=class_dir,
        pre_s=args.pre_s,
        post_s=args.post_s,
        selected_class=args.class_b,
    )

    if not np.allclose(common_t_a, common_t_b):
        raise RuntimeError("Time axes differ between class matrices.")

    common_t = common_t_a
    mat_a = first_a if args.tone_group == "first3" else last_a
    mat_b = first_b if args.tone_group == "first3" else last_b

    mean_a = np.nanmean(mat_a, axis=0)
    mean_b = np.nanmean(mat_b, axis=0)
    sem_a = np.nanstd(mat_a, axis=0, ddof=1) / np.sqrt(mat_a.shape[0]) if mat_a.shape[0] > 1 else np.full(mean_a.shape, np.nan)
    sem_b = np.nanstd(mat_b, axis=0, ddof=1) / np.sqrt(mat_b.shape[0]) if mat_b.shape[0] > 1 else np.full(mean_b.shape, np.nan)

    psth_df = pd.DataFrame(
        {
            "time_s": common_t,
            f"mean_{args.class_a}": mean_a,
            f"sem_{args.class_a}": sem_a,
            f"mean_{args.class_b}": mean_b,
            f"sem_{args.class_b}": sem_b,
        }
    )
    psth_df.to_csv(output_dir / f"{prefix}_PSTH_meanSEM.csv", index=False)

    per_bin_rows = []
    for idx, time_s in enumerate(common_t):
        res = compare_independent(mat_a[:, idx], mat_b[:, idx])
        res["time_s"] = float(time_s)
        per_bin_rows.append(res)
    per_bin_df = pd.DataFrame(per_bin_rows)
    add_fdr(per_bin_df, "mannwhitney_p", "mannwhitney_p_fdr_bh", "mannwhitney_sig_fdr_bh_0p05")
    add_fdr(per_bin_df, "welch_p", "welch_p_fdr_bh", "welch_sig_fdr_bh_0p05")
    per_bin_df.to_csv(output_dir / f"{prefix}_per_bin_stats.csv", index=False)

    pre_mask = (common_t >= -5.0) & (common_t < 0.0)
    post_mask = (common_t >= 0.0) & (common_t <= 3.0)
    post_t = common_t[post_mask]

    valid_a_pre = np.all(np.isfinite(mat_a[:, pre_mask]), axis=1)
    valid_b_pre = np.all(np.isfinite(mat_b[:, pre_mask]), axis=1)
    valid_a_post = np.all(np.isfinite(mat_a[:, post_mask]), axis=1)
    valid_b_post = np.all(np.isfinite(mat_b[:, post_mask]), axis=1)

    pre_a = np.nanmean(mat_a[:, pre_mask], axis=1)[valid_a_pre]
    pre_b = np.nanmean(mat_b[:, pre_mask], axis=1)[valid_b_pre]
    post_a = np.nanmean(mat_a[:, post_mask], axis=1)[valid_a_post]
    post_b = np.nanmean(mat_b[:, post_mask], axis=1)[valid_b_post]
    delta_a = (np.nanmean(mat_a[:, post_mask], axis=1) - np.nanmean(mat_a[:, pre_mask], axis=1))[valid_a_pre & valid_a_post]
    delta_b = (np.nanmean(mat_b[:, post_mask], axis=1) - np.nanmean(mat_b[:, pre_mask], axis=1))[valid_b_pre & valid_b_post]

    window_summary = pd.DataFrame(
        [
            {"comparison": "pre_-5to0", **compare_independent(pre_a, pre_b)},
            {"comparison": "post_0to3", **compare_independent(post_a, post_b)},
            {"comparison": "delta_(0to3_minus_-5to0)", **compare_independent(delta_a, delta_b)},
        ]
    )
    window_summary.to_csv(output_dir / f"{prefix}_window_summary.csv", index=False)

    auc_raw_a = np.trapezoid(mat_a[:, post_mask], x=post_t, axis=1)[valid_a_post]
    auc_raw_b = np.trapezoid(mat_b[:, post_mask], x=post_t, axis=1)[valid_b_post]
    pre_mean_a_all = np.nanmean(mat_a[:, pre_mask], axis=1)
    pre_mean_b_all = np.nanmean(mat_b[:, pre_mask], axis=1)
    auc_bc_a = np.trapezoid(mat_a[:, post_mask] - pre_mean_a_all[:, None], x=post_t, axis=1)[valid_a_pre & valid_a_post]
    auc_bc_b = np.trapezoid(mat_b[:, post_mask] - pre_mean_b_all[:, None], x=post_t, axis=1)[valid_b_pre & valid_b_post]
    auc_summary = pd.DataFrame(
        [
            {"comparison": "AUC_0to3_raw", **compare_independent(auc_raw_a, auc_raw_b)},
            {"comparison": "AUC_0to3_baselineCorrected", **compare_independent(auc_bc_a, auc_bc_b)},
        ]
    )
    auc_summary.to_csv(output_dir / f"{prefix}_AUC_summary.csv", index=False)

    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.25])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0.plot(common_t, mean_a, color="#1f77b4", linewidth=2.6, label=args.class_a)
    ax0.fill_between(common_t, mean_a - sem_a, mean_a + sem_a, color="#1f77b4", alpha=0.2)
    ax0.plot(common_t, mean_b, color="#ff7f0e", linewidth=2.6, label=args.class_b)
    ax0.fill_between(common_t, mean_b - sem_b, mean_b + sem_b, color="#ff7f0e", alpha=0.2)
    ax0.axvline(0, color="steelblue", alpha=0.45)
    ax0.axhline(0, color="steelblue", alpha=0.2)
    y_min = np.nanmin(np.r_[mean_a - sem_a, mean_b - sem_b])
    y_max = np.nanmax(np.r_[mean_a + sem_a, mean_b + sem_b])
    y_range = y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
    sig_mask = per_bin_df["mannwhitney_sig_fdr_bh_0p05"].to_numpy(dtype=bool)
    marker_y = y_min - 0.08 * y_range
    if sig_mask.any():
        ax0.scatter(common_t[sig_mask], np.full(sig_mask.sum(), marker_y), s=14, marker="s", color="black", label="Mann-Whitney FDR < 0.05")
    ax0.set_ylim(marker_y - 0.05 * y_range, y_max + 0.08 * y_range)
    ax0.set_xlabel("Time from tone onset (s)")
    ax0.set_ylabel("Delta z from onset")
    ax0.set_title(f"{tone_label(args.tone_group)} PSTH and per-bin significance")
    ax0.legend(frameon=False, loc="upper right")

    def paired_strip(ax, data_a, data_b, title, ylabel):
        x0, x1 = 0, 1
        ax.scatter(np.full(len(data_a), x0), data_a, color="#1f77b4", alpha=0.55, s=18)
        ax.scatter(np.full(len(data_b), x1), data_b, color="#ff7f0e", alpha=0.55, s=18)
        sem_left = np.std(data_a, ddof=1) / np.sqrt(len(data_a)) if len(data_a) > 1 else np.nan
        sem_right = np.std(data_b, ddof=1) / np.sqrt(len(data_b)) if len(data_b) > 1 else np.nan
        ax.errorbar([x0, x1], [np.mean(data_a), np.mean(data_b)], yerr=[sem_left, sem_right], fmt="k_", markersize=22, linewidth=2.2)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels([args.class_a, args.class_b])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="steelblue", alpha=0.2)

    paired_strip(ax1, auc_bc_a, auc_bc_b, "Baseline-corrected AUC 0-3 s", "AUC (Delta z * s)")
    paired_strip(ax2, delta_a, delta_b, "Window delta: (0-3 s) - (-5-0 s)", "Post-pre mean Delta z")

    fig.suptitle(
        f"Ext2 {args.class_a} vs {args.class_b} at {tone_label(args.tone_group)}\nUpdated 2/3, 0.5z, 3 s criteria",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_summary_panel.png", dpi=300)
    plt.close(fig)

    meta = {
        "classification_source": str(class_dir / "Ext2_cellClassifications_long.csv"),
        "criteria": {
            "thr_z": 0.5,
            "consec_frames": 10,
            "window_s": 3.0,
            "min_hits_per_period": 2,
            "early_tones": [1, 2, 3],
            "late_tones": [10, 11, 12],
        },
        "tone_group": args.tone_group,
        "tone_group_label": tone_label(args.tone_group),
        "class_a": args.class_a,
        "class_b": args.class_b,
        "trace_window_s": {"pre": args.pre_s, "post": args.post_s},
        "per_bin_primary_test": "two-sided Mann-Whitney U",
        "per_bin_secondary_test": "Welch t-test",
        "multiple_comparisons": "Benjamini-Hochberg FDR across time bins",
        "n_cells_class_a": int(len(cells_a)),
        "n_cells_class_b": int(len(cells_b)),
        "n_sig_bins_mannwhitney_fdr_bh_0p05": int(sig_mask.sum()),
    }
    (output_dir / f"{prefix}_metadata.json").write_text(json.dumps(meta, indent=2))

    print("Wrote outputs to", output_dir)
    print("n_cells_class_a", len(cells_a))
    print("n_cells_class_b", len(cells_b))
    print("sig_bins", int(sig_mask.sum()))


if __name__ == "__main__":
    main()
