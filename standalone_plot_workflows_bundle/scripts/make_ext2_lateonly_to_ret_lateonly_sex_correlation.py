#!/usr/bin/env python3
"""Plot Ext2 LateOnly -> Ret LateOnly matched-cell AUC correlations split by sex."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "matched_cell_crossed_session_analysis_2of3_3s" / "ext2_lateonly_to_ret_lateonly_sex_correlation"
PAIR_FILE = (
    ROOT
    / "ext2_ret_transition_output_updatedCriteria_2of3_3s/registered_pair_auc_validation/Ext2_LateOnly_to_Ret_LateOnly_registered_pairs_withAUC.csv"
)


def normalize_animal_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    return s


def normalize_cell_id(value: object) -> str:
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"C{int(digits):03d}" if digits else s


def load_sex_map() -> dict[str, str]:
    sex_map: dict[str, str] = {}
    candidates = [
        ROOT / "Ret_classes_proportions_perAnimal.csv",
        ROOT / "ext1_pipeline_output_2of3_3s_fullregen/Ext1_classes_proportions_perAnimal.csv",
        ROOT / "ext2_pipeline_output_clean_3s2of3/ext2_classes/Ext2_classes_proportions_perAnimal.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "animal_id" not in df.columns or "sex" not in df.columns:
            continue
        for _, row in df[["animal_id", "sex"]].dropna().iterrows():
            animal = normalize_animal_id(row["animal_id"])
            sex_map[animal] = str(row["sex"]).strip()
    return sex_map


def corr_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    out = {
        "n": int(len(x)),
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
    }
    if len(x) >= 2:
        pr, pp = stats.pearsonr(x, y)
        out["pearson_r"] = float(pr)
        out["pearson_p"] = float(pp)
    if len(x) >= 3:
        sr, sp = stats.spearmanr(x, y)
        out["spearman_rho"] = float(sr)
        out["spearman_p"] = float(sp)
    return out


def plot_panel(ax, df: pd.DataFrame, title: str, color: str, level: str) -> dict[str, float]:
    x = df["source_auc"].to_numpy(dtype=float)
    y = df["target_auc"].to_numpy(dtype=float)
    ax.scatter(x, y, s=52 if level == "animal_mean" else 38, color=color, alpha=0.82, edgecolor="white", linewidth=0.5)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() >= 2:
        lo = float(min(np.nanmin(x[finite]), np.nanmin(y[finite])))
        hi = float(max(np.nanmax(x[finite]), np.nanmax(y[finite])))
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="#8c8c8c", linewidth=1)
        if finite.sum() >= 2 and np.unique(x[finite]).size >= 2:
            coeffs = np.polyfit(x[finite], y[finite], 1)
            xp = np.linspace(lo - pad, hi + pad, 100)
            ax.plot(xp, coeffs[0] * xp + coeffs[1], color="#333333", linewidth=1.4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    stats_out = corr_stats(x, y)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Ext2 LateOnly AUC (0-3 s, baseline-corrected)", fontsize=9)
    ax.set_ylabel("Ret LateOnly AUC (0-3 s, baseline-corrected)", fontsize=9)
    ann = (
        f"n={stats_out['n']}\n"
        f"Spearman rho={stats_out['spearman_rho']:.3f}\n"
        f"p={stats_out['spearman_p']:.3g}\n"
        f"Pearson r={stats_out['pearson_r']:.3f}\n"
        f"p={stats_out['pearson_p']:.3g}"
    )
    ax.text(
        0.03,
        0.97,
        ann,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )
    ax.grid(alpha=0.24)
    return stats_out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    sub = pd.read_csv(PAIR_FILE).copy()
    sub["animal_id"] = sub["animal_id"].map(normalize_animal_id)
    sub["cell_id"] = sub["cell_id"].map(normalize_cell_id)
    sub = sub[sub["valid_both_auc"].fillna(False)].copy()
    sub = sub.rename(columns={"ext2_bc_auc": "source_auc", "ret_bc_auc": "target_auc"})
    sub = sub[["animal_id", "cell_id", "source_auc", "target_auc"]].copy()
    sub["target_minus_source"] = sub["target_auc"] - sub["source_auc"]
    sub["sex"] = sub["animal_id"].map(load_sex_map())
    sub = sub[sub["sex"].isin(["Female", "Male"])].copy()

    animal = (
        sub.groupby(["animal_id", "sex"], as_index=False)[["source_auc", "target_auc"]]
        .mean(numeric_only=True)
        .sort_values(["sex", "animal_id"])
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6), squeeze=False)
    palette = {"Female": "#c05a6d", "Male": "#4472c4"}
    stats_rows = []
    for col, sex in enumerate(["Female", "Male"]):
        cell_sub = sub[sub["sex"] == sex].copy()
        animal_sub = animal[animal["sex"] == sex].copy()
        cell_stats = plot_panel(axes[0, col], cell_sub, f"{sex} cell-level", palette[sex], "cell")
        animal_stats = plot_panel(axes[1, col], animal_sub, f"{sex} animal-mean", palette[sex], "animal_mean")
        stats_rows.append({"sex": sex, "level": "cell", **cell_stats, "n_animals": int(cell_sub["animal_id"].nunique())})
        stats_rows.append({"sex": sex, "level": "animal_mean", **animal_stats, "n_animals": int(animal_sub["animal_id"].nunique())})

    fig.suptitle("Ext2 LateOnly -> Ret LateOnly AUC Correlations by Sex", fontsize=13, y=0.995)
    fig.tight_layout()
    png_path = OUTDIR / "Ext2_LateOnly_to_Ret_LateOnly_AUC_correlations_bySex.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    pairs_csv = OUTDIR / "Ext2_LateOnly_to_Ret_LateOnly_pairs_bySex.csv"
    stats_csv = OUTDIR / "Ext2_LateOnly_to_Ret_LateOnly_correlation_stats_bySex.csv"
    sub.sort_values(["sex", "animal_id", "cell_id"]).to_csv(pairs_csv, index=False)
    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)

    meta = {
        "comparison": "Ext2 LateOnly -> Ret LateOnly",
        "source": str(PAIR_FILE),
        "notes": [
            "Pairs come from the explicit Ext2-to-Ret registered-pair export, then are filtered to rows with valid AUC in both sessions.",
            "This is the strict LateOnly-to-LateOnly matched-cell overlap, not the all-registered comparison.",
            "Top row is cell-level and is descriptive because cells are nested within animals.",
            "Bottom row averages matched cells within animal before computing the correlation.",
        ],
    }
    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
