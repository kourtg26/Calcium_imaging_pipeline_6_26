#!/usr/bin/env python3
"""Plot paired animal-level tone-responsive retention proportions across transitions."""

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
OUTDIR = (
    ROOT
    / "matched_cell_crossed_session_analysis_2of3_3s"
    / "ext1ext2_vs_ext2ret_toneResponsive_retentionProp_pairedTtests"
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


def subset_stats(df: pd.DataFrame) -> dict[str, float]:
    x = df["ext1_ext2_prop_retained_resp"].to_numpy(dtype=float)
    y = df["ext2_ret_prop_retained_resp"].to_numpy(dtype=float)
    t_stat, p_val, n = paired_t(x, y)
    return {
        "n": n,
        "mean_x": float(np.nanmean(x)) if n else np.nan,
        "mean_y": float(np.nanmean(y)) if n else np.nan,
        "sem_x": sem(x),
        "sem_y": sem(y),
        "t": t_stat,
        "p": p_val,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PAIR_FILE).copy()
    df = df[df["sex"].isin(["Female", "Male"])].copy()
    df = df.sort_values(["sex", "animal_id"]).reset_index(drop=True)

    panels = [
        ("All", df.copy(), "#6c757d", "#222222"),
        ("Female", df[df["sex"] == "Female"].copy(), "#c05a6d", "#7c2f3f"),
        ("Male", df[df["sex"] == "Male"].copy(), "#4472c4", "#23457e"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    xpos = np.array([0.0, 1.0])

    summary_rows = []
    excluded_rows = []

    for ax, (label, sub_all, line_color, point_color) in zip(axes, panels):
        valid = sub_all[
            sub_all["ext1_ext2_prop_retained_resp"].notna() & sub_all["ext2_ret_prop_retained_resp"].notna()
        ].copy()
        excluded = sub_all[
            sub_all["ext1_ext2_prop_retained_resp"].isna() | sub_all["ext2_ret_prop_retained_resp"].isna()
        ].copy()

        for _, row in valid.iterrows():
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
                s=34,
                color=point_color,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
                zorder=2,
            )

        stats_row = subset_stats(valid)
        ax.errorbar(
            xpos,
            [stats_row["mean_x"], stats_row["mean_y"]],
            yerr=[stats_row["sem_x"], stats_row["sem_y"]],
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
        ax.grid(axis="y", alpha=0.22)
        ax.set_ylim(-0.03, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ann = (
            f"n={stats_row['n']} paired animals\n"
            f"mean E1->E2={stats_row['mean_x']:.3f}\n"
            f"mean E2->R={stats_row['mean_y']:.3f}\n"
            f"paired t={stats_row['t']:.3f}\n"
            f"p={stats_row['p']:.3g}"
        )
        if not excluded.empty:
            ann += f"\nexcluded={len(excluded)}"
        ax.text(
            0.03,
            0.97,
            ann,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
        )

        summary_rows.append(
            {
                "subset": label,
                "n_paired_animals": stats_row["n"],
                "n_excluded_animals": int(len(excluded)),
                "mean_ext1_ext2_prop_retained_resp": stats_row["mean_x"],
                "sem_ext1_ext2_prop_retained_resp": stats_row["sem_x"],
                "mean_ext2_ret_prop_retained_resp": stats_row["mean_y"],
                "sem_ext2_ret_prop_retained_resp": stats_row["sem_y"],
                "paired_t_stat": stats_row["t"],
                "paired_t_p": stats_row["p"],
            }
        )
        if not excluded.empty:
            for _, row in excluded.iterrows():
                excluded_rows.append(
                    {
                        "subset": label,
                        "animal_id": row["animal_id"],
                        "sex": row["sex"],
                        "ext1_ext2_prop_retained_resp": row["ext1_ext2_prop_retained_resp"],
                        "ext2_ret_prop_retained_resp": row["ext2_ret_prop_retained_resp"],
                    }
                )

    axes[0].set_ylabel("Retention proportion of tone-responsive cells")
    fig.suptitle("Paired Animal Retention Proportions Across Transitions", y=1.03)
    fig.tight_layout()

    png_path = OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests.png"
    pdf_path = OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(summary_rows).to_csv(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_plotSummary.csv",
        index=False,
    )
    pd.DataFrame(excluded_rows).to_csv(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_excludedAnimals.csv",
        index=False,
    )

    with open(OUTDIR / "paired_ttest_plot_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "analysis": "Paired animal-level tone-responsive retention proportion plot",
                "source": str(PAIR_FILE),
                "outputs": {
                    "png": str(png_path),
                    "pdf": str(pdf_path),
                },
                "notes": [
                    "Each line is one paired animal.",
                    "Panels show all animals, females only, and males only.",
                    "Animals with missing retention proportion in either transition are excluded from the paired t-test panel calculations.",
                ],
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
