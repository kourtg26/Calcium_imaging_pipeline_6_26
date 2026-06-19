#!/usr/bin/env python3
"""Plot male vs female tone-responsive retention proportions across transitions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXT1_EXT2_FILE = ROOT / "ext1_ext2_transition_output_updatedCriteria_2of3_3s/Ext1toExt2_transition_toneResponsive_byAnimal.csv"
EXT2_RET_FILE = ROOT / "ext2_ret_transition_output_updatedCriteria_2of3_3s/Ext2toRet_transition_toneResponsive_byAnimal.csv"
EXT1_EXT2_STATS = ROOT / "ext1_ext2_transition_output_updatedCriteria_2of3_3s/Ext1toExt2_transition_sexDiff_toneResponsive_retention_stats.csv"
EXT2_RET_STATS = ROOT / "ext2_ret_transition_output_updatedCriteria_2of3_3s/Ext2toRet_transition_sexDiff_toneResponsive_retention_stats.csv"
OUTDIR = ROOT / "matched_cell_crossed_session_analysis_2of3_3s" / "toneResponsive_retentionProp_sexComparison_plot"


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


def load_transition_table(path: Path, transition: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_id"] = df["animal"].map(normalize_animal_id)
    df["sex"] = df["sex"].astype(str).str.strip()
    df["transition"] = transition
    return df[["animal_id", "sex", "transition", "prop_retained_resp"]].copy()


def load_stat_p(path: Path) -> float:
    df = pd.read_csv(path)
    row = df.loc[df["metric"] == "prop_retained_resp"].iloc[0]
    return float(row["welch_p"])


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [
            load_transition_table(EXT1_EXT2_FILE, "Ext1 -> Ext2"),
            load_transition_table(EXT2_RET_FILE, "Ext2 -> Ret"),
        ],
        ignore_index=True,
    )
    combined = combined[combined["sex"].isin(["Female", "Male"])].copy()
    combined = combined.sort_values(["transition", "sex", "animal_id"]).reset_index(drop=True)

    pvals = {
        "Ext1 -> Ext2": load_stat_p(EXT1_EXT2_STATS),
        "Ext2 -> Ret": load_stat_p(EXT2_RET_STATS),
    }

    summary = (
        combined.groupby(["transition", "sex"], as_index=False)["prop_retained_resp"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    colors = {"Female": "#c05a6d", "Male": "#4472c4"}
    transitions = ["Ext1 -> Ext2", "Ext2 -> Ret"]
    sexes = ["Female", "Male"]
    x_centers = np.array([0.0, 1.5])
    offsets = {"Female": -0.18, "Male": 0.18}

    fig, ax = plt.subplots(figsize=(8.4, 6.2))

    for transition_idx, transition in enumerate(transitions):
        for sex in sexes:
            sub = combined[(combined["transition"] == transition) & (combined["sex"] == sex)].copy()
            x0 = x_centers[transition_idx] + offsets[sex]
            if not sub.empty:
                jitter = np.linspace(-0.045, 0.045, len(sub)) if len(sub) > 1 else np.array([0.0])
                ax.scatter(
                    np.full(len(sub), x0) + jitter,
                    sub["prop_retained_resp"].to_numpy(dtype=float),
                    s=58,
                    color=colors[sex],
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )
                mean_y = float(sub["prop_retained_resp"].mean())
                sem_y = sem(sub["prop_retained_resp"].to_numpy(dtype=float))
                ax.errorbar(
                    [x0],
                    [mean_y],
                    yerr=[sem_y],
                    fmt="_",
                    markersize=24,
                    color="#111111",
                    ecolor="#111111",
                    elinewidth=1.8,
                    capsize=4,
                    capthick=1.8,
                    zorder=4,
                )

        y_ann = min(
            1.03,
            float(
                combined.loc[combined["transition"] == transition, "prop_retained_resp"].max()
            )
            + 0.12,
        )
        x_left = x_centers[transition_idx] + offsets["Female"]
        x_right = x_centers[transition_idx] + offsets["Male"]
        ax.plot([x_left, x_left, x_right, x_right], [y_ann - 0.02, y_ann, y_ann, y_ann - 0.02], color="#333333", lw=1.2)
        ax.text(
            x_centers[transition_idx],
            y_ann + 0.015,
            f"Welch p = {pvals[transition]:.3g}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x_centers, transitions)
    ax.set_ylabel("Retention proportion of tone-responsive cells")
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Male vs Female Tone-Responsive Retention")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[sex], markeredgecolor="white", markersize=9)
        for sex in sexes
    ]
    ax.legend(legend_handles, sexes, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTDIR / "toneResponsive_retentionProp_sexComparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "toneResponsive_retentionProp_sexComparison.pdf", bbox_inches="tight")
    plt.close(fig)

    combined.to_csv(OUTDIR / "toneResponsive_retentionProp_sexComparison_plotValues.csv", index=False)
    summary.to_csv(OUTDIR / "toneResponsive_retentionProp_sexComparison_summary.csv", index=False)

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "analysis": "Male vs female tone-responsive retention proportions across transitions",
                "transitions": transitions,
                "sources": {
                    "ext1_ext2_table": str(EXT1_EXT2_FILE),
                    "ext2_ret_table": str(EXT2_RET_FILE),
                    "ext1_ext2_stats": str(EXT1_EXT2_STATS),
                    "ext2_ret_stats": str(EXT2_RET_STATS),
                },
                "notes": [
                    "Points are individual animals.",
                    "Black horizontal markers show mean +/- SEM.",
                    "Annotation uses the existing Welch sex-comparison p-values for each transition.",
                ],
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
