#!/usr/bin/env python3
"""Matched-cell crossed-session analyses for early/late switching subsets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "matched_cell_crossed_session_analysis_2of3_3s"


def normalize_animal_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    return s


def bh_fdr(pvals: list[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    pv = p[mask]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    restored = np.empty_like(adj)
    restored[order] = adj
    out[mask] = restored
    return out


def cohen_dz(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(diff) / sd)


def paired_wilcoxon(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.nan, np.nan, 0
    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0, len(x)
    stat, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
    return float(stat), float(p), int(len(x))


def paired_ttest(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, len(x)
    res = stats.ttest_rel(x, y, nan_policy="omit")
    return float(res.statistic), float(res.pvalue), int(len(x))


def load_auc_table(path: Path, session: str, class_label: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_id"] = df["animal_id"].map(normalize_animal_id)
    df["cell_id"] = df["cell_id"].astype(str)
    df = df[df["valid_auc_test"]].copy()
    df = df[["animal_id", "cell_id", value_col]].rename(columns={value_col: "auc_value"})
    df["session"] = session
    df["cell_class"] = class_label
    return df


def load_explicit_registered_pair_table(
    path: Path, comparison: str, source_session: str, source_class: str, target_session: str, target_class: str
) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_id"] = df["animal_id"].map(normalize_animal_id)
    df["cell_id"] = df["cell_id"].astype(str)
    df = df[df["valid_both_auc"].fillna(False)].copy()
    df = df.rename(columns={"ext2_bc_auc": "source_auc", "ret_bc_auc": "target_auc"})
    df = df[["animal_id", "cell_id", "source_auc", "target_auc"]].copy()
    df["comparison"] = comparison
    df["source_session"] = source_session
    df["source_class"] = source_class
    df["target_session"] = target_session
    df["target_class"] = target_class
    df["target_minus_source"] = df["target_auc"] - df["source_auc"]
    return df


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    comparisons = [
        {
            "comparison": "Ext1 EarlyOnly -> Ext2 LateOnly",
            "source_session": "Ext1",
            "target_session": "Ext2",
            "source_class": "EarlyOnly",
            "target_class": "LateOnly",
            "source_file": ROOT
            / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "target_file": ROOT
            / "ext2_psth_lateOnly_allCells_first3_vs_last3_updatedCriteria_2of3_3s/Ext2_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "source_value_col": "first_auc_0to3_baselineCorrected",
            "target_value_col": "last_auc_0to3_baselineCorrected",
        },
        {
            "comparison": "Ext1 LateOnly -> Ext2 EarlyOnly",
            "source_session": "Ext1",
            "target_session": "Ext2",
            "source_class": "LateOnly",
            "target_class": "EarlyOnly",
            "source_file": ROOT
            / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "target_file": ROOT
            / "ext2_psth_earlyOnly_allCells_first3_vs_last3_updatedCriteria_2of3_3s/Ext2_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "source_value_col": "last_auc_0to3_baselineCorrected",
            "target_value_col": "first_auc_0to3_baselineCorrected",
        },
        {
            "comparison": "Ext2 EarlyOnly -> Ret LateOnly",
            "source_session": "Ext2",
            "target_session": "Ret",
            "source_class": "EarlyOnly",
            "target_class": "LateOnly",
            "pair_file": ROOT
            / "ext2_ret_transition_output_updatedCriteria_2of3_3s/registered_pair_auc_validation/Ext2_EarlyOnly_to_Ret_LateOnly_registered_pairs_withAUC.csv",
        },
        {
            "comparison": "Ext2 LateOnly -> Ret EarlyOnly",
            "source_session": "Ext2",
            "target_session": "Ret",
            "source_class": "LateOnly",
            "target_class": "EarlyOnly",
            "pair_file": ROOT
            / "ext2_ret_transition_output_updatedCriteria_2of3_3s/registered_pair_auc_validation/Ext2_LateOnly_to_Ret_EarlyOnly_registered_pairs_withAUC.csv",
        },
        {
            "comparison": "Ext1 EarlyOnly -> Ret LateOnly",
            "source_session": "Ext1",
            "target_session": "Ret",
            "source_class": "EarlyOnly",
            "target_class": "LateOnly",
            "source_file": ROOT
            / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "target_file": ROOT
            / "ret_per_cell_auc_tone1_vs_tone4_updatedCriteria_2of3_3s/Ret_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "source_value_col": "first_auc_0to3_baselineCorrected",
            "target_value_col": "tone4_auc_0to3_baselineCorrected",
        },
        {
            "comparison": "Ext1 LateOnly -> Ret EarlyOnly",
            "source_session": "Ext1",
            "target_session": "Ret",
            "source_class": "LateOnly",
            "target_class": "EarlyOnly",
            "source_file": ROOT
            / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "target_file": ROOT
            / "ret_per_cell_auc_tone1_vs_tone4_updatedCriteria_2of3_3s/Ret_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
            "source_value_col": "last_auc_0to3_baselineCorrected",
            "target_value_col": "tone1_auc_0to3_baselineCorrected",
        },
    ]

    pair_rows = []
    summary_rows = []
    for spec in comparisons:
        if "pair_file" in spec:
            paired = load_explicit_registered_pair_table(
                spec["pair_file"],
                spec["comparison"],
                spec["source_session"],
                spec["source_class"],
                spec["target_session"],
                spec["target_class"],
            )
        else:
            src = load_auc_table(
                spec["source_file"], spec["source_session"], spec["source_class"], spec["source_value_col"]
            ).rename(columns={"auc_value": "source_auc"})
            tgt = load_auc_table(
                spec["target_file"], spec["target_session"], spec["target_class"], spec["target_value_col"]
            ).rename(columns={"auc_value": "target_auc"})
            paired = src.merge(tgt, on=["animal_id", "cell_id"], how="inner", suffixes=("_src", "_tgt"))
            paired["comparison"] = spec["comparison"]
            paired = paired.rename(
                columns={
                    "session_src": "source_session",
                    "cell_class_src": "source_class",
                    "session_tgt": "target_session",
                    "cell_class_tgt": "target_class",
                }
            )
            paired["target_minus_source"] = paired["target_auc"] - paired["source_auc"]
        pair_rows.append(
            paired[
                [
                    "comparison",
                    "animal_id",
                    "cell_id",
                    "source_session",
                    "source_class",
                    "source_auc",
                    "target_session",
                    "target_class",
                    "target_auc",
                    "target_minus_source",
                ]
            ]
        )

        x = paired["source_auc"].to_numpy(dtype=float)
        y = paired["target_auc"].to_numpy(dtype=float)
        diff = y - x
        wil_stat, wil_p, n = paired_wilcoxon(x, y)
        t_stat, t_p, _ = paired_ttest(x, y)
        rho, rho_p = (np.nan, np.nan)
        if n >= 3:
            rho, rho_p = stats.spearmanr(x, y)
        summary_rows.append(
            {
                "comparison": spec["comparison"],
                "source_session": spec["source_session"],
                "target_session": spec["target_session"],
                "source_class": spec["source_class"],
                "target_class": spec["target_class"],
                "n_cells": n,
                "n_animals": paired["animal_id"].nunique(),
                "source_mean": float(np.nanmean(x)) if n else np.nan,
                "target_mean": float(np.nanmean(y)) if n else np.nan,
                "mean_target_minus_source": float(np.nanmean(diff)) if n else np.nan,
                "median_target_minus_source": float(np.nanmedian(diff)) if n else np.nan,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(rho_p) if np.isfinite(rho_p) else np.nan,
                "wilcoxon_stat": wil_stat,
                "wilcoxon_p": wil_p,
                "paired_t_stat": t_stat,
                "paired_t_p": t_p,
                "cohen_dz_target_minus_source": cohen_dz(diff),
            }
        )

    pairs_df = pd.concat(pair_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["wilcoxon_p_fdr_bh"] = bh_fdr(summary_df["wilcoxon_p"].tolist())
    summary_df["spearman_p_fdr_bh"] = bh_fdr(summary_df["spearman_p"].tolist())
    summary_df["paired_t_p_fdr_bh"] = bh_fdr(summary_df["paired_t_p"].tolist())
    summary_df = summary_df.sort_values("wilcoxon_p", kind="stable")

    pairs_df.to_csv(OUTDIR / "matched_cell_crossed_session_pairs.csv", index=False)
    summary_df.to_csv(OUTDIR / "matched_cell_crossed_session_summary.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(10, 13), squeeze=False)
    axes_flat = axes.flatten()
    for ax, spec in zip(axes_flat, comparisons):
        sub = pairs_df[pairs_df["comparison"] == spec["comparison"]].copy()
        row = summary_df.loc[summary_df["comparison"] == spec["comparison"]].iloc[0]
        x = sub["source_auc"].to_numpy(dtype=float)
        y = sub["target_auc"].to_numpy(dtype=float)
        ax.scatter(x, y, s=45, color="#1f4e79", alpha=0.85)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 2:
            lo = float(min(np.nanmin(x[finite]), np.nanmin(y[finite])))
            hi = float(max(np.nanmax(x[finite]), np.nanmax(y[finite])))
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#999999", linewidth=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(spec["comparison"], fontsize=10)
        ax.set_xlabel(f"{spec['source_session']} {spec['source_class']} crossed-source AUC", fontsize=9)
        ax.set_ylabel(f"{spec['target_session']} {spec['target_class']} crossed-target AUC", fontsize=9)
        ann = (
            f"cells={int(row['n_cells'])}, animals={int(row['n_animals'])}\n"
            f"mean delta={row['mean_target_minus_source']:.3f}\n"
            f"Wilcoxon p={row['wilcoxon_p']:.3g}\n"
            f"FDR={row['wilcoxon_p_fdr_bh']:.3g}"
        )
        ax.text(
            0.03,
            0.97,
            ann,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
        ax.grid(alpha=0.25)
    fig.suptitle("Matched-Cell Crossed-Session Early/Late Transition Comparisons", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(OUTDIR / "matched_cell_crossed_session_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "criteria": {
            "thr_z": 0.5,
            "consec_frames": 10,
            "window_s": 3.0,
            "extinction_min_hits_per_period": 2,
            "extinction_early_tones": [1, 2, 3],
            "extinction_late_tones": [10, 11, 12],
            "retrieval_early_tone": 1,
            "retrieval_late_tone": 4,
        },
        "notes": [
            "Most comparisons are matched by normalized animal_id plus cell_id, using session AUC tables derived from registered session traces.",
            "Ext2 EarlyOnly -> Ret LateOnly is sourced from the explicit Ext2-to-Ret registered-pair export filtered to rows with valid AUC in both sessions.",
            "Ext2 LateOnly -> Ret EarlyOnly is sourced from the explicit Ext2-to-Ret registered-pair export filtered to rows with valid AUC in both sessions.",
            "Crossed-session comparisons only include cells that are present in both class-specific per-cell AUC tables for the comparison.",
            "Target minus source is the primary paired difference direction in the summary table.",
            "FDR is applied across the six crossed-session comparisons.",
        ],
    }
    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


if __name__ == "__main__":
    main()
