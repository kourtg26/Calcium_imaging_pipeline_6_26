#!/usr/bin/env python3
"""Paired t-tests and paired-mean plots for Ext2 LateOnly -> Ret EarlyOnly AUC."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
PAIR_FILE = (
    ROOT
    / "ext2_ret_transition_output_updatedCriteria_2of3_3s/registered_pair_auc_validation/Ext2_LateOnly_to_Ret_EarlyOnly_registered_pairs_withAUC.csv"
)
OUTDIR = ROOT / "matched_cell_crossed_session_analysis_2of3_3s" / "ext2_lateonly_to_ret_earlyonly_paired_ttest_meanplot"


def normalize_animal_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    return s


def normalize_cell_id(value: object) -> str:
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"C{int(digits):03d}" if digits else s


def sem(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PAIR_FILE).copy()
    df["animal_id"] = df["animal_id"].map(normalize_animal_id)
    df["cell_id"] = df["cell_id"].map(normalize_cell_id)
    df = df[df["valid_both_auc"].fillna(False)].copy()
    df = df.rename(
        columns={
            "ext2_bc_auc": "ext2_late_auc_bc",
            "ret_bc_auc": "ret_early_auc_bc",
            "ext2_raw_auc": "ext2_late_auc_raw",
            "ret_raw_auc": "ret_early_auc_raw",
        }
    )
    df = df[
        [
            "animal_id",
            "cell_id",
            "ext2_late_auc_bc",
            "ret_early_auc_bc",
            "ext2_late_auc_raw",
            "ret_early_auc_raw",
        ]
    ].copy()
    df["delta_ret_minus_ext2_bc"] = df["ret_early_auc_bc"] - df["ext2_late_auc_bc"]
    df["delta_ret_minus_ext2_raw"] = df["ret_early_auc_raw"] - df["ext2_late_auc_raw"]
    df = df.sort_values(["animal_id", "cell_id"]).reset_index(drop=True)

    n = int(len(df))
    metrics = [
        (
            "baseline_corrected_auc_0to3",
            "ext2_late_auc_bc",
            "ret_early_auc_bc",
            "delta_ret_minus_ext2_bc",
            "Baseline-corrected AUC (0-3 s)",
        ),
        (
            "raw_auc_0to3",
            "ext2_late_auc_raw",
            "ret_early_auc_raw",
            "delta_ret_minus_ext2_raw",
            "Raw AUC (0-3 s)",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), squeeze=False)
    summary_rows = []
    xpos = np.array([0.0, 1.0])
    for ax, (metric_name, src_col, tgt_col, delta_col, ylabel) in zip(axes[0], metrics):
        x = df[src_col].to_numpy(dtype=float)
        y = df[tgt_col].to_numpy(dtype=float)
        t_res = stats.ttest_rel(x, y, nan_policy="omit")
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        sem_x = sem(x)
        sem_y = sem(y)
        mean_delta = float(np.mean(df[delta_col].to_numpy(dtype=float)))

        for _, row in df.iterrows():
            ax.plot(
                xpos,
                [row[src_col], row[tgt_col]],
                color="#b8b8b8",
                linewidth=1.0,
                alpha=0.8,
                zorder=1,
            )
            ax.scatter(
                xpos,
                [row[src_col], row[tgt_col]],
                s=34,
                color=["#2f6db2", "#c2545d"],
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )

        ax.errorbar(
            xpos,
            [mean_x, mean_y],
            yerr=[sem_x, sem_y],
            fmt="o",
            markersize=9,
            color="#111111",
            ecolor="#111111",
            elinewidth=1.6,
            capsize=4,
            capthick=1.6,
            zorder=3,
        )

        ax.set_xticks(xpos, ["Ext2 LateOnly", "Ret EarlyOnly"])
        ax.set_ylabel(ylabel)
        ax.set_title(metric_name.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.22)

        ann = (
            f"n={n} cells\n"
            f"mean Ext2={mean_x:.3f}\n"
            f"mean Ret={mean_y:.3f}\n"
            f"mean delta={mean_delta:.3f}\n"
            f"paired t={t_res.statistic:.3f}\n"
            f"p={t_res.pvalue:.3g}"
        )
        ax.text(
            0.04,
            0.97,
            ann,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )

        summary_rows.append(
            {
                "comparison": "Ext2 LateOnly -> Ret EarlyOnly",
                "metric": metric_name,
                "n_cells": n,
                "mean_ext2_late_auc": mean_x,
                "sem_ext2_late_auc": sem_x,
                "mean_ret_early_auc": mean_y,
                "sem_ret_early_auc": sem_y,
                "mean_delta_ret_minus_ext2": mean_delta,
                "paired_t_stat": float(t_res.statistic),
                "paired_t_p": float(t_res.pvalue),
            }
        )

    fig.suptitle("Ext2 LateOnly -> Ret EarlyOnly\nPaired Cell AUC Means", y=1.02)
    fig.tight_layout()
    png_path = OUTDIR / "Ext2_LateOnly_to_Ret_EarlyOnly_paired_ttest_meanplot.png"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(OUTDIR / "Ext2_LateOnly_to_Ret_EarlyOnly_paired_ttest_values.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "Ext2_LateOnly_to_Ret_EarlyOnly_paired_ttest_summary.csv", index=False)

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "comparison": "Ext2 LateOnly -> Ret EarlyOnly",
                "metrics": ["baseline_corrected_auc_0to3", "raw_auc_0to3"],
                "source": str(PAIR_FILE),
                "notes": [
                    "Pairs come from the explicit Ext2-to-Ret registered-pair export.",
                    "Only rows with valid AUC in both sessions are included.",
                    "The plot shows one line per matched cell plus mean +/- SEM for both baseline-corrected and raw AUC.",
                ],
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
