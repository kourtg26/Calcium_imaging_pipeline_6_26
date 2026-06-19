#!/usr/bin/env python3
"""Median-split learner-group paired tests for tone-responsive retention proportions."""

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
    / "matched_cell_crossed_session_analysis_2of3_3s"
    / "ext1ext2_vs_ext2ret_toneResponsive_retentionProp_pairedTtests"
    / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_byAnimal.csv"
)
BEHAVIOR_FILE = ROOT / "ext2_behavior_index_correlations_early_late" / "per_animal_behavior_index_and_cell_activity_metrics.csv"
OUTDIR = (
    ROOT
    / "matched_cell_crossed_session_analysis_2of3_3s"
    / "ext1ext2_vs_ext2ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit"
)


def normalize_animal_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    return s


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


def learner_summary(df: pd.DataFrame, label: str) -> dict[str, object]:
    x = df["ext1_ext2_prop_retained_resp"].to_numpy(dtype=float)
    y = df["ext2_ret_prop_retained_resp"].to_numpy(dtype=float)
    t_stat, p_val, n = paired_t(x, y)
    return {
        "learner_group": label,
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

    pair = pd.read_csv(PAIR_FILE).copy()
    behavior = pd.read_csv(BEHAVIOR_FILE).copy()
    behavior["animal_id"] = behavior["animal"].map(normalize_animal_id)

    merged = pair.merge(
        behavior[["animal_id", "frz_tone1_mean", "frz_tone12_mean", "behavior_index_last_minus_first"]],
        on="animal_id",
        how="left",
    ).copy()

    valid = merged[
        merged["ext1_ext2_prop_retained_resp"].notna()
        & merged["ext2_ret_prop_retained_resp"].notna()
        & merged["behavior_index_last_minus_first"].notna()
    ].copy()

    median_behavior = float(valid["behavior_index_last_minus_first"].median())
    valid["learner_group"] = np.where(
        valid["behavior_index_last_minus_first"] <= median_behavior,
        "Good learners",
        "Bad learners",
    )
    valid = valid.sort_values(["learner_group", "behavior_index_last_minus_first", "animal_id"]).reset_index(drop=True)

    groups = [
        ("Good learners", "#2f7d5b", "#174c37"),
        ("Bad learners", "#b65e3c", "#7a351a"),
    ]

    summary_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=True)
    xpos = np.array([0.0, 1.0])

    for ax, (label, line_color, point_color) in zip(axes, groups):
        sub = valid[valid["learner_group"] == label].copy()
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
                s=36,
                color=point_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.92,
                zorder=2,
            )

        stats_row = learner_summary(sub, label)
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
        ax.set_title(label)
        ax.set_ylim(-0.03, 1.08)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.03,
            0.97,
            (
                f"n={stats_row['n_animals']} animals\n"
                f"median split cutoff={median_behavior:.3f}\n"
                f"mean behavior idx={stats_row['mean_behavior_index_last_minus_first']:.3f}\n"
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

    axes[0].set_ylabel("Retention proportion of tone-responsive cells")
    fig.suptitle(
        "Paired Animal Retention by Learner Group\n"
        "Good learners = lower Ext2 behavior index (tone12 freezing - tone1 freezing)",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit.pdf", bbox_inches="tight")
    plt.close(fig)

    valid.to_csv(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_byAnimal.csv",
        index=False,
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_learnerMedianSplit_summary.csv",
        index=False,
    )

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "analysis": "Median-split learner-group paired t-tests for tone-responsive retention proportions",
                "behavior_metric": "behavior_index_last_minus_first = freezing(Ext2 tone12) - freezing(Ext2 tone1)",
                "median_cutoff": median_behavior,
                "good_learner_rule": "behavior_index_last_minus_first <= median_cutoff",
                "bad_learner_rule": "behavior_index_last_minus_first > median_cutoff",
                "sources": {
                    "paired_retention": str(PAIR_FILE),
                    "ext2_behavior_index": str(BEHAVIOR_FILE),
                },
                "notes": [
                    "Lower behavior index means greater reduction in freezing across Ext2 and is treated as better extinction learning.",
                    "Median split is computed within animals that have finite paired retention values and finite behavior index.",
                ],
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
