#!/usr/bin/env python3
"""Build animal-level correlation summaries across Ext1, Ext2, and retrieval."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "session_correlation_workflow_ext1_ext2_ret_2of3_3s"
EXT1_FREEZE_EXCLUDE_ANIMALS = {"animal1", "animal2"}


def normalize_animal_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    return s


def bh_fdr(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    p_valid = p[mask]
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[mask] = restored
    return out


def load_class_props(path: Path, session: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_norm"] = df["animal_id"].map(normalize_animal_id)
    rename = {
        "n_cells": f"{session}_n_cells",
        "n_cells_total": f"{session}_n_cells",
        "n_EarlyOnly": f"{session}_n_EarlyOnly",
        "n_LateOnly": f"{session}_n_LateOnly",
        "p_EarlyOnly": f"{session}_p_EarlyOnly",
        "p_LateOnly": f"{session}_p_LateOnly",
        "prop_EarlyOnly": f"{session}_p_EarlyOnly",
        "prop_LateOnly": f"{session}_p_LateOnly",
    }
    keep = ["animal_norm", "sex"] + [c for c in rename if c in df.columns]
    df = df[keep].rename(columns=rename)
    if "sex" in df.columns:
        df = df.rename(columns={"sex": f"{session}_sex"})
    return df


def choose_best_ret_class_props() -> Path:
    candidates = [
        ROOT / "retrieval_pipeline_outputs_bundle/ret_classes_proportions_only/Ret_classes_proportions_perAnimal.csv",
        ROOT / "retrieval_pipeline_outputs_bundle/ret_classes_proportions_only 2/Ret_classes_proportions_perAnimal.csv",
        ROOT / "Ret_classes_proportions_perAnimal.csv",
        ROOT / "ret_classes_proportions_only/Ret_classes_proportions_perAnimal.csv",
    ]
    best_path = None
    best_score = (-1, -1)
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "animal_id" not in df.columns:
            continue
        score = (df["animal_id"].nunique(), len(df))
        if score > best_score:
            best_path = path
            best_score = score
    if best_path is None:
        raise FileNotFoundError("No retrieval class summary CSV found")
    return best_path


def load_freezing(path: Path, session: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_norm"] = df["animal_id"].map(normalize_animal_id)
    rename = {}
    for col in df.columns:
        if col == "animal_norm":
            continue
        rename[col] = f"{session}_{col}"
    keep = ["animal_norm"] + [c for c in df.columns if c != "animal_norm"]
    return df[keep].rename(columns=rename)


def aggregate_ext2_per_cell(path: Path, value_col: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_norm"] = df["animal_id"].map(normalize_animal_id)
    out = (
        df[df["valid_auc_test"]]
        .groupby("animal_norm", as_index=False)
        .agg(
            **{
                f"{prefix}_mean": (value_col, "mean"),
                f"{prefix}_median": (value_col, "median"),
                f"{prefix}_n_cells": (value_col, "size"),
            }
        )
    )
    return out


def aggregate_ext2_by_class_from_allcells(
    auc_path: Path, class_long_path: Path, class_label: str, value_col: str, prefix: str
) -> pd.DataFrame:
    auc = pd.read_csv(auc_path).copy()
    auc["animal_norm"] = auc["animal_id"].map(normalize_animal_id)
    auc["cell_norm"] = auc["cell_id"].astype(str).str.strip()

    cls = pd.read_csv(class_long_path).copy()
    cls["animal_norm"] = cls["animal_id"].map(normalize_animal_id)
    cls["cell_norm"] = cls["cell_id"].astype(str).str.strip()
    class_col = "cell_class" if "cell_class" in cls.columns else "class"

    merged = auc.merge(cls[["animal_norm", "cell_norm", class_col]], on=["animal_norm", "cell_norm"], how="inner")
    valid = merged[merged[class_col] == class_label].copy()
    valid = valid[valid["valid_auc_test"]].copy()
    return (
        valid.groupby("animal_norm", as_index=False)
        .agg(
            **{
                f"{prefix}_mean": (value_col, "mean"),
                f"{prefix}_median": (value_col, "median"),
                f"{prefix}_n_cells": (value_col, "size"),
            }
        )
    )


def aggregate_per_cell_file(path: Path, value_col: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_norm"] = df["animal_id"].map(normalize_animal_id)
    valid = df[df["valid_auc_test"]].copy()
    return (
        valid.groupby("animal_norm", as_index=False)
        .agg(
            **{
                f"{prefix}_mean": (value_col, "mean"),
                f"{prefix}_median": (value_col, "median"),
                f"{prefix}_n_cells": (value_col, "size"),
            }
        )
    )


def compute_corr(df: pd.DataFrame, x_col: str, y_col: str, family: str, label: str) -> dict:
    sub = df[["animal_norm", x_col, y_col]].dropna()
    n = len(sub)
    if n < 3:
        return {
            "family": family,
            "comparison": label,
            "x_col": x_col,
            "y_col": y_col,
            "n_animals": n,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }
    rho, rho_p = spearmanr(sub[x_col], sub[y_col])
    r, r_p = pearsonr(sub[x_col], sub[y_col])
    return {
        "family": family,
        "comparison": label,
        "x_col": x_col,
        "y_col": y_col,
        "n_animals": n,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
    }


def build_coverage_rows(df: pd.DataFrame, comparisons: list[tuple[str, str, str, str]]) -> list[dict]:
    all_animals = sorted(df["animal_norm"].dropna().astype(str).unique().tolist())
    rows = []
    for family, label, x_col, y_col in comparisons:
        sub = df[["animal_norm", x_col, y_col]].dropna()
        included = sorted(sub["animal_norm"].astype(str).unique().tolist())
        excluded = [animal for animal in all_animals if animal not in included]
        rows.append(
            {
                "family": family,
                "comparison": label,
                "x_col": x_col,
                "y_col": y_col,
                "n_animals": int(len(included)),
                "included_animals": ";".join(included),
                "excluded_from_master_animals": ";".join(excluded),
            }
        )
    return rows


def make_scatter_panel(
    df: pd.DataFrame,
    results: pd.DataFrame,
    comparisons: list[str],
    title: str,
    outpath: Path,
) -> None:
    n = len(comparisons)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax, comp in zip(axes_flat, comparisons):
        row = results.loc[results["comparison"] == comp].iloc[0]
        sub = df[["animal_norm", row["x_col"], row["y_col"]]].dropna()
        ax.scatter(sub[row["x_col"]], sub[row["y_col"]], s=45, color="#1f4e79", alpha=0.85)
        if len(sub) >= 2:
            x = sub[row["x_col"]].to_numpy(dtype=float)
            y = sub[row["y_col"]].to_numpy(dtype=float)
            coeffs = np.polyfit(x, y, 1)
            xp = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yp = coeffs[0] * xp + coeffs[1]
            ax.plot(xp, yp, color="#c84c09", linewidth=1.5)
        ax.set_title(comp, fontsize=10)
        ax.set_xlabel(row["x_col"], fontsize=9)
        ax.set_ylabel(row["y_col"], fontsize=9)
        ann = (
            f"n={int(row['n_animals'])}\n"
            f"Spearman rho={row['spearman_rho']:.3f}\n"
            f"p={row['spearman_p']:.3g}\n"
            f"FDR={row['spearman_p_fdr_bh_all']:.3g}"
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
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ext1_class = load_class_props(
        ROOT / "ext1_pipeline_output_2of3_3s_fullregen/Ext1_classes_proportions_perAnimal.csv",
        "Ext1",
    )
    ext2_class = load_class_props(
        ROOT / "ext2_pipeline_output_clean_3s2of3/ext2_classes/Ext2_classes_proportions_perAnimal.csv",
        "Ext2",
    )
    ret_class = load_class_props(choose_best_ret_class_props(), "Ret")

    ext1_freeze = load_freezing(
        ROOT / "ext1_pipeline_output_2of3_3s_fullregen/Ext1_toneBlock_activity_freezing_perAnimal.csv",
        "Ext1",
    )
    ext2_freeze = load_freezing(
        ROOT / "ext2_pipeline_output_clean_3s2of3/Ext2_toneBlock_activity_freezing_perAnimal.csv",
        "Ext2",
    )

    ext1_early_auc = aggregate_per_cell_file(
        ROOT
        / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
        "first_auc_0to3_baselineCorrected",
        "Ext1_Early_pref_bc_auc_0to3",
    )
    ext1_late_auc = aggregate_per_cell_file(
        ROOT
        / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
        "last_auc_0to3_baselineCorrected",
        "Ext1_Late_pref_bc_auc_0to3",
    )
    ret_early_auc = aggregate_per_cell_file(
        ROOT
        / "ret_per_cell_auc_tone1_vs_tone4_updatedCriteria_2of3_3s/Ret_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
        "tone1_auc_0to3_baselineCorrected",
        "Ret_Early_pref_bc_auc_0to3",
    )
    ret_late_auc = aggregate_per_cell_file(
        ROOT
        / "ret_per_cell_auc_tone1_vs_tone4_updatedCriteria_2of3_3s/Ret_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv",
        "tone4_auc_0to3_baselineCorrected",
        "Ret_Late_pref_bc_auc_0to3",
    )

    ext2_all_auc_path = (
        ROOT
        / "ext2_allCells_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext2_allCells_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    )
    ext2_class_long_path = (
        ROOT / "ext2_pipeline_output_clean_3s2of3/ext2_classes/Ext2_cellClassifications_long.csv"
    )

    ext2_early_auc = aggregate_ext2_by_class_from_allcells(
        ext2_all_auc_path,
        ext2_class_long_path,
        "EarlyOnly",
        "first_auc_0to3_baselineCorrected",
        "Ext2_Early_pref_bc_auc_0to3",
    )
    ext2_late_auc = aggregate_ext2_by_class_from_allcells(
        ext2_all_auc_path,
        ext2_class_long_path,
        "LateOnly",
        "last_auc_0to3_baselineCorrected",
        "Ext2_Late_pref_bc_auc_0to3",
    )

    overlap_e1e2 = pd.read_csv(
        ROOT / "ext1_ext2_transition_output_2of3_3s_fullregen/Ext1toExt2_overlap_diagnostics_idAlignment.csv"
    )
    overlap_e1e2["animal_norm"] = overlap_e1e2["animal_id"].map(normalize_animal_id)
    overlap_e1e2 = overlap_e1e2[["animal_norm", "n_registered"]].rename(
        columns={"n_registered": "Ext1toExt2_n_registered"}
    )

    overlap_e2r = pd.read_csv(
        ROOT / "ext2_ret_transition_output_2of3_3s_fullregen/Ext2toRet_overlap_diagnostics_idAlignment.csv"
    )
    overlap_e2r["animal_norm"] = overlap_e2r["animal_id"].map(normalize_animal_id)
    overlap_e2r = overlap_e2r[["animal_norm", "n_registered"]].rename(
        columns={"n_registered": "Ext2toRet_n_registered"}
    )

    merged = ext1_class.copy()
    for part in [
        ext2_class,
        ret_class,
        ext1_freeze,
        ext2_freeze,
        ext1_early_auc,
        ext1_late_auc,
        ret_early_auc,
        ret_late_auc,
        ext2_early_auc,
        ext2_late_auc,
        overlap_e1e2,
        overlap_e2r,
    ]:
        merged = merged.merge(part, on="animal_norm", how="outer")

    merged = merged[merged["animal_norm"].notna()].copy()
    merged = merged[merged["animal_norm"].astype(str).str.lower() != "ret"].copy()

    ext1_freeze_cols = [c for c in merged.columns if c.startswith("Ext1_freeze_")]
    if ext1_freeze_cols:
        mask = merged["animal_norm"].isin(EXT1_FREEZE_EXCLUDE_ANIMALS)
        merged.loc[mask, ext1_freeze_cols] = np.nan

    # Session-level bias indices summarize preference on a single axis.
    merged["Ext1_bias_bc_auc_mean"] = (
        merged["Ext1_Early_pref_bc_auc_0to3_mean"] - merged["Ext1_Late_pref_bc_auc_0to3_mean"]
    )
    merged["Ext2_bias_bc_auc_mean"] = (
        merged["Ext2_Early_pref_bc_auc_0to3_mean"] - merged["Ext2_Late_pref_bc_auc_0to3_mean"]
    )
    merged["Ret_bias_bc_auc_mean"] = (
        merged["Ret_Early_pref_bc_auc_0to3_mean"] - merged["Ret_Late_pref_bc_auc_0to3_mean"]
    )

    comparisons = [
        (
            "Within-session neural vs freezing",
            "Ext1 early preferred AUC vs early freezing",
            "Ext1_Early_pref_bc_auc_0to3_mean",
            "Ext1_freeze_block_1-3",
        ),
        (
            "Within-session neural vs freezing",
            "Ext1 late preferred AUC vs late freezing",
            "Ext1_Late_pref_bc_auc_0to3_mean",
            "Ext1_freeze_block_10-12",
        ),
        (
            "Within-session neural vs freezing",
            "Ext2 EarlyOnly preferred AUC vs early freezing",
            "Ext2_Early_pref_bc_auc_0to3_mean",
            "Ext2_freeze_block_1-3",
        ),
        (
            "Within-session neural vs freezing",
            "Ext2 LateOnly preferred AUC vs late freezing",
            "Ext2_Late_pref_bc_auc_0to3_mean",
            "Ext2_freeze_block_10-12",
        ),
        (
            "Cross-session class proportion stability",
            "EarlyOnly proportion Ext1 vs Ext2",
            "Ext1_p_EarlyOnly",
            "Ext2_p_EarlyOnly",
        ),
        (
            "Cross-session class proportion stability",
            "LateOnly proportion Ext1 vs Ext2",
            "Ext1_p_LateOnly",
            "Ext2_p_LateOnly",
        ),
        (
            "Cross-session class proportion stability",
            "EarlyOnly proportion Ext2 vs Ret",
            "Ext2_p_EarlyOnly",
            "Ret_p_EarlyOnly",
        ),
        (
            "Cross-session class proportion stability",
            "LateOnly proportion Ext2 vs Ret",
            "Ext2_p_LateOnly",
            "Ret_p_LateOnly",
        ),
        (
            "Cross-session class proportion stability",
            "EarlyOnly proportion Ext1 vs Ret",
            "Ext1_p_EarlyOnly",
            "Ret_p_EarlyOnly",
        ),
        (
            "Cross-session class proportion stability",
            "LateOnly proportion Ext1 vs Ret",
            "Ext1_p_LateOnly",
            "Ret_p_LateOnly",
        ),
        (
            "Cross-session preferred-tone neural stability",
            "Early preferred AUC Ext1 vs Ext2",
            "Ext1_Early_pref_bc_auc_0to3_mean",
            "Ext2_Early_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session preferred-tone neural stability",
            "Late preferred AUC Ext1 vs Ext2",
            "Ext1_Late_pref_bc_auc_0to3_mean",
            "Ext2_Late_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session preferred-tone neural stability",
            "Early preferred AUC Ext2 vs Ret",
            "Ext2_Early_pref_bc_auc_0to3_mean",
            "Ret_Early_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session preferred-tone neural stability",
            "Late preferred AUC Ext2 vs Ret",
            "Ext2_Late_pref_bc_auc_0to3_mean",
            "Ret_Late_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session preferred-tone neural stability",
            "Early preferred AUC Ext1 vs Ret",
            "Ext1_Early_pref_bc_auc_0to3_mean",
            "Ret_Early_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session preferred-tone neural stability",
            "Late preferred AUC Ext1 vs Ret",
            "Ext1_Late_pref_bc_auc_0to3_mean",
            "Ret_Late_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session crossed-tone neural remapping",
            "Early preferred AUC Ext1 vs Late preferred AUC Ext2",
            "Ext1_Early_pref_bc_auc_0to3_mean",
            "Ext2_Late_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session crossed-tone neural remapping",
            "Late preferred AUC Ext1 vs Early preferred AUC Ext2",
            "Ext1_Late_pref_bc_auc_0to3_mean",
            "Ext2_Early_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session crossed-tone neural remapping",
            "Early preferred AUC Ext2 vs Late preferred AUC Ret",
            "Ext2_Early_pref_bc_auc_0to3_mean",
            "Ret_Late_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session crossed-tone neural remapping",
            "Late preferred AUC Ext2 vs Early preferred AUC Ret",
            "Ext2_Late_pref_bc_auc_0to3_mean",
            "Ret_Early_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session crossed-tone neural remapping",
            "Early preferred AUC Ext1 vs Late preferred AUC Ret",
            "Ext1_Early_pref_bc_auc_0to3_mean",
            "Ret_Late_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session crossed-tone neural remapping",
            "Late preferred AUC Ext1 vs Early preferred AUC Ret",
            "Ext1_Late_pref_bc_auc_0to3_mean",
            "Ret_Early_pref_bc_auc_0to3_mean",
        ),
        (
            "Cross-session bias stability",
            "Bias index Ext1 vs Ext2",
            "Ext1_bias_bc_auc_mean",
            "Ext2_bias_bc_auc_mean",
        ),
        (
            "Cross-session bias stability",
            "Bias index Ext2 vs Ret",
            "Ext2_bias_bc_auc_mean",
            "Ret_bias_bc_auc_mean",
        ),
        (
            "Cross-session bias stability",
            "Bias index Ext1 vs Ret",
            "Ext1_bias_bc_auc_mean",
            "Ret_bias_bc_auc_mean",
        ),
    ]

    results = pd.DataFrame(
        [compute_corr(merged, x, y, family, label) for family, label, x, y in comparisons]
    )
    results["spearman_p_fdr_bh_all"] = bh_fdr(results["spearman_p"])
    results["pearson_p_fdr_bh_all"] = bh_fdr(results["pearson_p"])
    results = results.sort_values(["family", "spearman_p", "comparison"], kind="stable")

    coverage = pd.DataFrame(build_coverage_rows(merged, comparisons))

    merged.to_csv(OUTDIR / "session_correlation_inputs_by_animal.csv", index=False)
    results.to_csv(OUTDIR / "session_correlation_results_master.csv", index=False)
    coverage.to_csv(OUTDIR / "session_correlation_plot_coverage.csv", index=False)

    make_scatter_panel(
        merged,
        results,
        [
            "Ext1 early preferred AUC vs early freezing",
            "Ext1 late preferred AUC vs late freezing",
            "Ext2 EarlyOnly preferred AUC vs early freezing",
            "Ext2 LateOnly preferred AUC vs late freezing",
        ],
        "Within-Session Neural vs Freezing Correlations",
        OUTDIR / "within_session_neural_vs_freezing_correlations.png",
    )
    make_scatter_panel(
        merged,
        results,
        [
            "EarlyOnly proportion Ext1 vs Ext2",
            "LateOnly proportion Ext1 vs Ext2",
            "EarlyOnly proportion Ext2 vs Ret",
            "LateOnly proportion Ext2 vs Ret",
            "EarlyOnly proportion Ext1 vs Ret",
            "LateOnly proportion Ext1 vs Ret",
        ],
        "Cross-Session Class Proportion Stability",
        OUTDIR / "cross_session_class_proportion_correlations.png",
    )
    make_scatter_panel(
        merged,
        results,
        [
            "Early preferred AUC Ext1 vs Ext2",
            "Late preferred AUC Ext1 vs Ext2",
            "Early preferred AUC Ext2 vs Ret",
            "Late preferred AUC Ext2 vs Ret",
            "Early preferred AUC Ext1 vs Ret",
            "Late preferred AUC Ext1 vs Ret",
        ],
        "Cross-Session Preferred-Tone Neural Stability",
        OUTDIR / "cross_session_preferred_tone_auc_correlations.png",
    )
    make_scatter_panel(
        merged,
        results,
        [
            "Early preferred AUC Ext1 vs Late preferred AUC Ext2",
            "Late preferred AUC Ext1 vs Early preferred AUC Ext2",
            "Early preferred AUC Ext2 vs Late preferred AUC Ret",
            "Late preferred AUC Ext2 vs Early preferred AUC Ret",
            "Early preferred AUC Ext1 vs Late preferred AUC Ret",
            "Late preferred AUC Ext1 vs Early preferred AUC Ret",
        ],
        "Cross-Session Crossed-Tone Neural Remapping",
        OUTDIR / "cross_session_crossed_tone_auc_correlations.png",
    )
    make_scatter_panel(
        merged,
        results,
        [
            "Bias index Ext1 vs Ext2",
            "Bias index Ext2 vs Ret",
            "Bias index Ext1 vs Ret",
        ],
        "Cross-Session Bias Index Stability",
        OUTDIR / "cross_session_bias_index_correlations.png",
    )

    metadata = {
        "criteria": {
            "thr_z": 0.5,
            "consec_frames": 10,
            "window_s": 3.0,
            "min_hits_per_period": 2,
            "early_tones_extinction": [1, 2, 3],
            "late_tones_extinction": [10, 11, 12],
        },
        "notes": [
            "Animal IDs were normalized so numeric IDs 1/2/3... map to animal1/animal2/animal3...",
            "Ext1 and retrieval preferred-tone AUCs were derived from per-animal mean trace exports.",
            "Ext2 preferred-tone AUCs were aggregated by animal from the previously generated per-cell AUC tables.",
            "Bias index is defined as mean early-preferred baseline-corrected AUC minus mean late-preferred baseline-corrected AUC.",
            "Spearman correlations are primary; Pearson values are included as secondary summaries.",
            "Benjamini-Hochberg FDR was applied across all correlations in the master table.",
            "Ext1 freezing correlations exclude animal1 and animal2 because freezing data for that day should not be used.",
            "A coverage CSV lists the included animals for every plotted comparison.",
        ],
        "files_used": {
            "ext1_class": str(
                ROOT / "ext1_pipeline_output_2of3_3s_fullregen/Ext1_classes_proportions_perAnimal.csv"
            ),
            "ext2_class": str(
                ROOT
                / "ext2_pipeline_output_clean_3s2of3/ext2_classes/Ext2_classes_proportions_perAnimal.csv"
            ),
            "ret_class": str(
                ROOT
                / "retrieval_pipeline_outputs_bundle/ret_classes_proportions_only/Ret_classes_proportions_perAnimal.csv"
            ),
            "ext1_freeze": str(
                ROOT / "ext1_pipeline_output_2of3_3s_fullregen/Ext1_toneBlock_activity_freezing_perAnimal.csv"
            ),
            "ext2_freeze": str(
                ROOT / "ext2_pipeline_output_clean_3s2of3/Ext2_toneBlock_activity_freezing_perAnimal.csv"
            ),
            "ext1_early_auc_per_cell": str(
                ROOT
                / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
            ),
            "ext1_late_auc_per_cell": str(
                ROOT
                / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s/Ext1_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
            ),
            "ret_early_auc_per_cell": str(
                ROOT
                / "ret_per_cell_auc_tone1_vs_tone4_updatedCriteria_2of3_3s/Ret_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
            ),
            "ret_late_auc_per_cell": str(
                ROOT
                / "ret_per_cell_auc_tone1_vs_tone4_updatedCriteria_2of3_3s/Ret_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
            ),
            "ext2_early_auc_per_cell": str(
                ext2_all_auc_path
            ),
            "ext2_late_auc_per_cell": str(
                ext2_all_auc_path
            ),
            "ext2_class_long": str(
                ext2_class_long_path
            ),
        },
    }
    with open(OUTDIR / "session_correlation_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


if __name__ == "__main__":
    main()
