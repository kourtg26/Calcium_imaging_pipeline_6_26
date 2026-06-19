#!/usr/bin/env python3
"""Paired retention tests split by learner group and sex."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
INPUT_FILE = (
    ROOT
    / "matched_cell_crossed_session_analysis_2of3_3s"
    / "ext1ext2_vs_ext2ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit"
    / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_byAnimal.csv"
)
META_FILE = (
    ROOT
    / "matched_cell_crossed_session_analysis_2of3_3s"
    / "ext1ext2_vs_ext2ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit"
    / "metadata.json"
)
OUTDIR = (
    ROOT
    / "matched_cell_crossed_session_analysis_2of3_3s"
    / "ext1ext2_vs_ext2ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_bySex"
)


def sem(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def paired_t(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan, int(x.size)
    res = stats.ttest_rel(x, y, nan_policy="omit")
    return float(res.statistic), float(res.pvalue), int(x.size)


def summarize(df: pd.DataFrame, learner_group: str, sex: str) -> dict[str, object]:
    x = df["ext1_ext2_prop_retained_resp"].to_numpy(dtype=float)
    y = df["ext2_ret_prop_retained_resp"].to_numpy(dtype=float)
    t_stat, p_val, n = paired_t(x, y)
    return {
        "learner_group": learner_group,
        "sex": sex,
        "n_animals": n,
        "mean_behavior_index_last_minus_first": float(df["behavior_index_last_minus_first"].mean()) if n else np.nan,
        "mean_ext1_ext2_prop_retained_resp": float(np.mean(x)) if n else np.nan,
        "sem_ext1_ext2_prop_retained_resp": sem(x),
        "mean_ext2_ret_prop_retained_resp": float(np.mean(y)) if n else np.nan,
        "sem_ext2_ret_prop_retained_resp": sem(y),
        "mean_delta_ext2ret_minus_ext1ext2": float(np.mean(y - x)) if n else np.nan,
        "paired_t_stat": t_stat,
        "paired_t_p": p_val,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE).copy()
    with open(META_FILE, encoding="utf-8") as fh:
        meta = json.load(fh)
    cutoff = float(meta["median_cutoff"])

    panel_specs = [
        ("Good learners", "Female", "#2f7d5b", "#d16a7e"),
        ("Good learners", "Male", "#2f7d5b", "#4472c4"),
        ("Bad learners", "Female", "#b65e3c", "#d16a7e"),
        ("Bad learners", "Male", "#b65e3c", "#4472c4"),
    ]

    summary_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), sharey=True)
    xpos = np.array([0.0, 1.0])

    for ax, (learner_group, sex, line_color, point_color) in zip(axes.ravel(), panel_specs):
        sub = df[(df["learner_group"] == learner_group) & (df["sex"] == sex)].copy()
        sub = sub.sort_values("behavior_index_last_minus_first").reset_index(drop=True)

        for _, row in sub.iterrows():
            ax.plot(
                xpos,
                [row["ext1_ext2_prop_retained_resp"], row["ext2_ret_prop_retained_resp"]],
                color=line_color,
                linewidth=1.2,
                alpha=0.45,
                zorder=1,
            )
            ax.scatter(
                xpos,
                [row["ext1_ext2_prop_retained_resp"], row["ext2_ret_prop_retained_resp"]],
                s=38,
                color=point_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
                zorder=2,
            )

        stats_row = summarize(sub, learner_group, sex)
        summary_rows.append(stats_row)

        ax.errorbar(
            xpos,
            [stats_row["mean_ext1_ext2_prop_retained_resp"], stats_row["mean_ext2_ret_prop_retained_resp"]],
            yerr=[stats_row["sem_ext1_ext2_prop_retained_resp"], stats_row["sem_ext2_ret_prop_retained_resp"]],
            fmt="o",
            markersize=9,
            color="#111111",
            ecolor="#111111",
            elinewidth=1.7,
            capsize=4,
            capthick=1.7,
            zorder=3,
        )

        ax.set_xticks(xpos, ["Ext1 -> Ext2", "Ext2 -> Ret"])
        ax.set_title(f"{learner_group}, {sex}")
        ax.set_ylim(-0.03, 1.08)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.03,
            0.97,
            (
                f"n={stats_row['n_animals']} animals\n"
                f"cutoff={cutoff:.3f}\n"
                f"mean idx={stats_row['mean_behavior_index_last_minus_first']:.3f}\n"
                f"mean E1->E2={stats_row['mean_ext1_ext2_prop_retained_resp']:.3f}\n"
                f"mean E2->R={stats_row['mean_ext2_ret_prop_retained_resp']:.3f}\n"
                f"paired t={stats_row['paired_t_stat']:.3f}\n"
                f"p={stats_row['paired_t_p']:.3g}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
        )

    axes[0, 0].set_ylabel("Retention proportion of tone-responsive cells")
    axes[1, 0].set_ylabel("Retention proportion of tone-responsive cells")
    fig.suptitle(
        "Paired Animal Retention by Learner Group and Sex\n"
        "Learner split uses Ext2 tone12 freezing - Ext2 tone1 freezing",
        y=1.02,
    )
    fig.tight_layout()

    fig.savefig(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_bySex.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_bySex.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    df.to_csv(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_bySex_byAnimal.csv",
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_bySex_summary.csv",
        index=False,
    )

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "analysis": "Paired retention tests split by learner group and sex",
                "source": str(INPUT_FILE),
                "median_cutoff": cutoff,
                "notes": [
                    "Learner labels are inherited from the overall median split using Ext2 tone12 freezing - Ext2 tone1 freezing.",
                    "Panels are then separated by sex within those learner groups.",
                    "Some panels have small n and should be interpreted cautiously.",
                ],
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
