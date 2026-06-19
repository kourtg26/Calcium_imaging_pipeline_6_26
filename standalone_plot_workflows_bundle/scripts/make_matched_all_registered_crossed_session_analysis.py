#!/usr/bin/env python3
"""Crossed-session matched-cell analyses using all registered cells."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "matched_all_registered_crossed_session_analysis_2of3_3s"


def normalize_animal_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    return s


def normalize_cell_id(value: object) -> str:
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"C{int(digits):03d}" if digits else s


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


def load_auc(path: Path, early_col: str, late_col: str, session: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_id"] = df["animal_id"].map(normalize_animal_id)
    df["cell_id"] = df["cell_id"].map(normalize_cell_id)
    df = df[df["valid_auc_test"]].copy()
    return df[["animal_id", "cell_id", early_col, late_col]].rename(
        columns={early_col: f"{session}_early_auc", late_col: f"{session}_late_auc"}
    )


def summarize_animal_level(pairs: pd.DataFrame) -> tuple[float, float, int]:
    if pairs.empty:
        return np.nan, np.nan, 0
    animal = (
        pairs.groupby("animal_id", as_index=False)[["source_auc", "target_auc"]].mean(numeric_only=True)
    )
    if len(animal) < 2:
        return np.nan, np.nan, len(animal)
    stat, p = paired_wilcoxon(animal["source_auc"].to_numpy(), animal["target_auc"].to_numpy())[:2]
    return stat, p, len(animal)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ext1 = load_auc(
        ROOT / "ext1_allCells_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_allCells_AUC_per_cell_updatedCriteria_2of3_3s.csv",
        "first_auc_0to3_baselineCorrected",
        "last_auc_0to3_baselineCorrected",
        "Ext1",
    )
    ext2 = load_auc(
        ROOT / "ext2_allCells_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext2_allCells_AUC_per_cell_updatedCriteria_2of3_3s.csv",
        "first_auc_0to3_baselineCorrected",
        "last_auc_0to3_baselineCorrected",
        "Ext2",
    )
    ret = load_auc(
        ROOT / "ret_allCells_auc_tone1_vs_tone4_updatedCriteria_2of3_3s/Ret_allCells_AUC_per_cell_updatedCriteria_2of3_3s.csv",
        "tone1_auc_0to3_baselineCorrected",
        "tone4_auc_0to3_baselineCorrected",
        "Ret",
    )

    ext1_ext2 = ext1.merge(ext2, on=["animal_id", "cell_id"], how="inner")
    ext2_ret = ext2.merge(ret, on=["animal_id", "cell_id"], how="inner")
    ext1_ret = ext1.merge(ret, on=["animal_id", "cell_id"], how="inner")

    comparisons = [
        ("Ext1 early vs Ext2 early", ext1_ext2, "Ext1_early_auc", "Ext2_early_auc"),
        ("Ext1 late vs Ext2 late", ext1_ext2, "Ext1_late_auc", "Ext2_late_auc"),
        ("Ext1 early vs Ext2 late", ext1_ext2, "Ext1_early_auc", "Ext2_late_auc"),
        ("Ext1 late vs Ext2 early", ext1_ext2, "Ext1_late_auc", "Ext2_early_auc"),
        ("Ext2 early vs Ret early", ext2_ret, "Ext2_early_auc", "Ret_early_auc"),
        ("Ext2 late vs Ret late", ext2_ret, "Ext2_late_auc", "Ret_late_auc"),
        ("Ext2 early vs Ret late", ext2_ret, "Ext2_early_auc", "Ret_late_auc"),
        ("Ext2 late vs Ret early", ext2_ret, "Ext2_late_auc", "Ret_early_auc"),
        ("Ext1 early vs Ret early", ext1_ret, "Ext1_early_auc", "Ret_early_auc"),
        ("Ext1 late vs Ret late", ext1_ret, "Ext1_late_auc", "Ret_late_auc"),
        ("Ext1 early vs Ret late", ext1_ret, "Ext1_early_auc", "Ret_late_auc"),
        ("Ext1 late vs Ret early", ext1_ret, "Ext1_late_auc", "Ret_early_auc"),
    ]

    pair_rows = []
    summary_rows = []
    for label, base, xcol, ycol in comparisons:
        pairs = base[["animal_id", "cell_id", xcol, ycol]].dropna().copy()
        pairs = pairs.rename(columns={xcol: "source_auc", ycol: "target_auc"})
        pairs["comparison"] = label
        pairs["target_minus_source"] = pairs["target_auc"] - pairs["source_auc"]
        pair_rows.append(pairs[["comparison", "animal_id", "cell_id", "source_auc", "target_auc", "target_minus_source"]])

        x = pairs["source_auc"].to_numpy(dtype=float)
        y = pairs["target_auc"].to_numpy(dtype=float)
        diff = y - x
        wil_stat, wil_p, n = paired_wilcoxon(x, y)
        t_stat, t_p, _ = paired_ttest(x, y)
        rho, rho_p = (np.nan, np.nan)
        if n >= 3:
            rho, rho_p = stats.spearmanr(x, y)
        animal_wil_stat, animal_wil_p, n_animals = summarize_animal_level(pairs)
        summary_rows.append(
            {
                "comparison": label,
                "n_cells": n,
                "n_animals": pairs["animal_id"].nunique(),
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
                "animal_mean_wilcoxon_stat": animal_wil_stat,
                "animal_mean_wilcoxon_p": animal_wil_p,
                "animal_mean_n_animals": n_animals,
            }
        )

    pairs_df = pd.concat(pair_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["wilcoxon_p_fdr_bh"] = bh_fdr(summary_df["wilcoxon_p"].tolist())
    summary_df["spearman_p_fdr_bh"] = bh_fdr(summary_df["spearman_p"].tolist())
    summary_df["paired_t_p_fdr_bh"] = bh_fdr(summary_df["paired_t_p"].tolist())
    summary_df["animal_mean_wilcoxon_p_fdr_bh"] = bh_fdr(summary_df["animal_mean_wilcoxon_p"].tolist())
    summary_df = summary_df.sort_values("wilcoxon_p", kind="stable")

    pairs_df.to_csv(OUTDIR / "matched_all_registered_crossed_session_pairs.csv", index=False)
    summary_df.to_csv(OUTDIR / "matched_all_registered_crossed_session_summary.csv", index=False)

    n_comp = len(comparisons)
    n_cols = 2
    n_rows = math.ceil(n_comp / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4.3 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n_comp:]:
        ax.axis("off")
    for ax, (label, _, _, _) in zip(axes_flat, comparisons):
        sub = pairs_df[pairs_df["comparison"] == label]
        row = summary_df.loc[summary_df["comparison"] == label].iloc[0]
        x = sub["source_auc"].to_numpy(dtype=float)
        y = sub["target_auc"].to_numpy(dtype=float)
        ax.scatter(x, y, s=12, color="#1f4e79", alpha=0.55)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 2:
            lo = float(min(np.nanmin(x[finite]), np.nanmin(y[finite])))
            hi = float(max(np.nanmax(x[finite]), np.nanmax(y[finite])))
            pad = 0.08 * (hi - lo if hi > lo else 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#999999", linewidth=1)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Source crossed AUC", fontsize=9)
        ax.set_ylabel("Target crossed AUC", fontsize=9)
        ann = (
            f"cells={int(row['n_cells'])}, animals={int(row['n_animals'])}\n"
            f"mean delta={row['mean_target_minus_source']:.3f}\n"
            f"cell p={row['wilcoxon_p']:.3g}\n"
            f"animal p={row['animal_mean_wilcoxon_p']:.3g}"
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
    fig.suptitle("Matched All-Registered Cell Crossed-Session Comparisons", fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(OUTDIR / "matched_all_registered_crossed_session_panel.png", dpi=220, bbox_inches="tight")
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
            "Matching is by normalized animal_id plus cell_id across all-cell AUC tables.",
            "This approximates all registered cells under the identity-aligned cell ID scheme used in the transition outputs.",
            "Primary tests are cell-level paired Wilcoxon and secondary animal-level paired Wilcoxon on per-animal means.",
        ],
    }
    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


if __name__ == "__main__":
    main()
