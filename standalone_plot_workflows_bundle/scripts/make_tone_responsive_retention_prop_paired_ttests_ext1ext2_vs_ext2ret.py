#!/usr/bin/env python3
"""Paired animal-level t-tests for tone-responsive retention proportions across transitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
EXT1_EXT2_FILE = ROOT / "ext1_ext2_transition_output_updatedCriteria_2of3_3s/Ext1toExt2_transition_toneResponsive_byAnimal.csv"
EXT2_RET_FILE = ROOT / "ext2_ret_transition_output_updatedCriteria_2of3_3s/Ext2toRet_transition_toneResponsive_byAnimal.csv"
OUTDIR = ROOT / "matched_cell_crossed_session_analysis_2of3_3s" / "ext1ext2_vs_ext2ret_toneResponsive_retentionProp_pairedTtests"


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


def paired_ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan, int(x.size)
    res = stats.ttest_rel(x, y, nan_policy="omit")
    return float(res.statistic), float(res.pvalue), int(x.size)


def load_table(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["animal_id"] = df["animal"].map(normalize_animal_id)
    df["sex"] = df["sex"].astype(str).str.strip()
    return df.rename(
        columns={
            "prop_retained_resp": f"{prefix}_prop_retained_resp",
            "prop_lost_resp": f"{prefix}_prop_lost_resp",
        }
    )[
        [
            "animal_id",
            "sex",
            f"{prefix}_prop_retained_resp",
            f"{prefix}_prop_lost_resp",
        ]
    ]


def summarize_subset(df: pd.DataFrame, subset_label: str) -> dict[str, object]:
    x = df["ext1_ext2_prop_retained_resp"].to_numpy(dtype=float)
    y = df["ext2_ret_prop_retained_resp"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    t_stat, p_value, n_animals = paired_ttest(x, y)
    delta = y - x
    return {
        "subset": subset_label,
        "n_animals": n_animals,
        "mean_ext1_ext2_prop_retained_resp": float(np.nanmean(x)) if n_animals else np.nan,
        "sem_ext1_ext2_prop_retained_resp": sem(x),
        "mean_ext2_ret_prop_retained_resp": float(np.nanmean(y)) if n_animals else np.nan,
        "sem_ext2_ret_prop_retained_resp": sem(y),
        "mean_delta_ext2ret_minus_ext1ext2": float(np.nanmean(delta)) if n_animals else np.nan,
        "paired_t_stat": t_stat,
        "paired_t_p": p_value,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ext1_ext2 = load_table(EXT1_EXT2_FILE, "ext1_ext2")
    ext2_ret = load_table(EXT2_RET_FILE, "ext2_ret")

    merged = ext1_ext2.merge(
        ext2_ret,
        on="animal_id",
        how="inner",
        suffixes=("_ext1ext2", "_ext2ret"),
    ).copy()
    merged["sex"] = merged["sex_ext1ext2"].where(
        merged["sex_ext1ext2"].isin(["Female", "Male"]),
        merged["sex_ext2ret"],
    )
    merged = merged[
        [
            "animal_id",
            "sex",
            "ext1_ext2_prop_retained_resp",
            "ext2_ret_prop_retained_resp",
            "ext1_ext2_prop_lost_resp",
            "ext2_ret_prop_lost_resp",
        ]
    ].copy()
    merged["delta_ext2ret_minus_ext1ext2"] = (
        merged["ext2_ret_prop_retained_resp"] - merged["ext1_ext2_prop_retained_resp"]
    )
    merged = merged.sort_values(["sex", "animal_id"]).reset_index(drop=True)

    summary_rows = [summarize_subset(merged, "All")]
    for sex in ["Female", "Male"]:
        summary_rows.append(summarize_subset(merged[merged["sex"] == sex].copy(), sex))

    merged.to_csv(OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_byAnimal.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        OUTDIR / "Ext1Ext2_vs_Ext2Ret_toneResponsive_retentionProp_pairedTtests_summary.csv",
        index=False,
    )

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "analysis": "Paired animal-level t-tests on tone-responsive retention proportions",
                "comparison": "Ext1->Ext2 versus Ext2->Ret",
                "metric": "prop_retained_resp",
                "sources": {
                    "ext1_ext2": str(EXT1_EXT2_FILE),
                    "ext2_ret": str(EXT2_RET_FILE),
                },
                "notes": [
                    "Animals are matched by normalized animal_id across the two transition tables.",
                    "The paired t-test compares each animal's Ext1->Ext2 retention proportion against its Ext2->Ret retention proportion.",
                    "Separate summaries are provided for all animals, females, and males.",
                ],
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
