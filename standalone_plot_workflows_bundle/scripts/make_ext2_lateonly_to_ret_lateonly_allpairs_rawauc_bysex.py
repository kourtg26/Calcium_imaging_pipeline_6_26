#!/usr/bin/env python3
"""Plot permissive raw AUC by sex for all Ext2 LateOnly -> Ret LateOnly registered pairs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from freezing_pipeline_all_code_bundle.utils_freezing import (
    choose_one_ext2_csv_per_animal,
    extract_csvs_from_zip,
    infer_fps,
    load_ext2_session_csv,
)


ROOT = Path(__file__).resolve().parent
PAIR_FILE = (
    ROOT
    / "ext2_ret_transition_output_updatedCriteria_2of3_3s/registered_pair_auc_validation/Ext2_LateOnly_to_Ret_LateOnly_registered_pairs_withAUC.csv"
)
OUTDIR = ROOT / "matched_cell_crossed_session_analysis_2of3_3s" / "ext2_lateonly_to_ret_lateonly_allpairs_rawauc_bySex"


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
            sex_map[normalize_animal_id(row["animal_id"])] = str(row["sex"]).strip()
    return sex_map


def tone_epochs_from_arrays(tone_id: np.ndarray | None, tone_flag: np.ndarray | None) -> list[dict]:
    if tone_id is not None:
        tid = pd.to_numeric(pd.Series(tone_id), errors="coerce").to_numpy()
        uniq = [int(x) for x in np.unique(tid[np.isfinite(tid)]) if int(x) != 0]
        if len(uniq) > 1:
            epochs = []
            for tone_num in sorted(uniq):
                idxs = np.where(tid == tone_num)[0]
                if len(idxs) == 0:
                    continue
                epochs.append({"tone_num": tone_num, "onset_idx": int(idxs[0]), "idxs": idxs})
            return sorted(epochs, key=lambda d: d["onset_idx"])

    if tone_flag is None:
        return []
    flag = pd.to_numeric(pd.Series(tone_flag), errors="coerce").fillna(0).astype(int).to_numpy()
    rises = np.where((flag[1:] == 1) & (flag[:-1] == 0))[0] + 1
    falls = np.where((flag[1:] == 0) & (flag[:-1] == 1))[0] + 1
    if len(flag) and flag[0] == 1:
        rises = np.r_[0, rises]
    if len(flag) and flag[-1] == 1:
        falls = np.r_[falls, len(flag)]
    return [
        {"tone_num": order, "onset_idx": int(start), "idxs": np.arange(start, stop)}
        for order, (start, stop) in enumerate(zip(rises, falls), start=1)
    ]


def sem(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def paired_t(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan
    res = stats.ttest_rel(x, y, nan_policy="omit")
    return float(res.statistic), float(res.pvalue)


def permissive_postonly_raw_auc(z: np.ndarray, onsets: list[int], cell_idx: int, fps: float, window_s: float = 3.0) -> tuple[float, int]:
    post_f = int(round(window_s * fps))
    traces = []
    for onset in onsets:
        if onset + post_f >= z.shape[0]:
            continue
        seg = z[onset : onset + post_f + 1, cell_idx].astype(float)
        seg = seg - seg[0]
        traces.append(seg)
    if not traces:
        return np.nan, 0
    mean_trace = np.nanmean(np.vstack(traces), axis=0)
    tvec = np.arange(0, post_f + 1) / fps
    valid = np.isfinite(mean_trace) & np.isfinite(tvec)
    if valid.sum() < 2:
        return np.nan, len(traces)
    return float(np.trapezoid(mean_trace[valid], tvec[valid])), len(traces)


def compute_ext2_permissive_raw_auc(pair_df: pd.DataFrame) -> pd.DataFrame:
    tmp_dir = ROOT / "_tmp_ext2_allcells_auc"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csvs = extract_csvs_from_zip(str(ROOT / "ext2_raw_trace_files.zip"), str(tmp_dir))
    chosen = {normalize_animal_id(Path(fp).name.split("_")[0]): fp for fp in choose_one_ext2_csv_per_animal(csvs)}

    rows = []
    for animal, sub in pair_df.groupby("animal_id"):
        fp = chosen.get(animal)
        if fp is None:
            for _, row in sub.iterrows():
                rows.append({"animal_id": animal, "cell_id": row["cell_id"], "ext2_raw_auc_postOnly": np.nan, "ext2_n_late_tones_used": 0})
            continue
        d = load_ext2_session_csv(fp)
        if d is None:
            continue
        z = np.asarray(d["z"], dtype=float)
        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        epochs = tone_epochs_from_arrays(d.get("tone_id"), d.get("tone_flag"))
        if len(epochs) < 12:
            continue
        last_onsets = [e["onset_idx"] for e in epochs[-3:]]
        cell_ids = [normalize_cell_id(c) for c in d["cell_ids"]]
        cell_lookup = {cid: idx for idx, cid in enumerate(cell_ids)}
        for _, row in sub.iterrows():
            idx = cell_lookup.get(row["cell_id"])
            if idx is None:
                rows.append({"animal_id": animal, "cell_id": row["cell_id"], "ext2_raw_auc_postOnly": np.nan, "ext2_n_late_tones_used": 0})
                continue
            auc, n_used = permissive_postonly_raw_auc(z, last_onsets, idx, fps)
            rows.append({"animal_id": animal, "cell_id": row["cell_id"], "ext2_raw_auc_postOnly": auc, "ext2_n_late_tones_used": n_used})
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(PAIR_FILE).copy()
    pairs["animal_id"] = pairs["animal_id"].map(normalize_animal_id)
    pairs["cell_id"] = pairs["cell_id"].map(normalize_cell_id)
    pairs["sex"] = pairs["animal_id"].map(load_sex_map())
    pairs = pairs[pairs["sex"].isin(["Female", "Male"])].copy()

    ext2_postonly = compute_ext2_permissive_raw_auc(pairs)
    df = pairs.merge(ext2_postonly, on=["animal_id", "cell_id"], how="left")
    df = df.rename(columns={"ret_raw_auc": "ret_late_raw_auc"})
    df["ext2_raw_auc_allPairs"] = df["ext2_raw_auc_postOnly"].where(
        np.isfinite(df["ext2_raw_auc_postOnly"]), df["ext2_raw_auc"]
    )
    df["delta_ret_minus_ext2_raw"] = df["ret_late_raw_auc"] - df["ext2_raw_auc_allPairs"]
    df = df.sort_values(["sex", "animal_id", "cell_id"]).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.9), squeeze=False)
    colors = {"Female": ("#c05a6d", "#d9929e"), "Male": ("#4472c4", "#89a8de")}
    summary_rows = []
    xpos = np.array([0.0, 1.0])

    for ax, sex in zip(axes[0], ["Female", "Male"]):
        sub = df[df["sex"] == sex].copy()
        x = sub["ext2_raw_auc_allPairs"].to_numpy(dtype=float)
        y = sub["ret_late_raw_auc"].to_numpy(dtype=float)
        t_stat, t_p = paired_t(x, y)
        mean_x = float(np.nanmean(x)) if len(sub) else np.nan
        mean_y = float(np.nanmean(y)) if len(sub) else np.nan
        sem_x = sem(x)
        sem_y = sem(y)
        mean_delta = float(np.nanmean(sub["delta_ret_minus_ext2_raw"].to_numpy(dtype=float))) if len(sub) else np.nan

        for _, row in sub.iterrows():
            ax.plot(xpos, [row["ext2_raw_auc_allPairs"], row["ret_late_raw_auc"]], color="#b8b8b8", linewidth=1.0, alpha=0.8, zorder=1)
            ax.scatter(
                xpos,
                [row["ext2_raw_auc_allPairs"], row["ret_late_raw_auc"]],
                s=34,
                color=[colors[sex][0], colors[sex][1]],
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
        ax.set_xticks(xpos, ["Ext2 LateOnly", "Ret LateOnly"])
        ax.set_ylabel("Raw AUC (0-3 s, all pairs)")
        ax.set_title(f"{sex}")
        ax.grid(axis="y", alpha=0.22)
        ann = (
            f"n={len(sub)} cells\n"
            f"animals={sub['animal_id'].nunique()}\n"
            f"mean Ext2={mean_x:.3f}\n"
            f"mean Ret={mean_y:.3f}\n"
            f"mean delta={mean_delta:.3f}\n"
            f"paired t={t_stat:.3f}\n"
            f"p={t_p:.3g}"
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
                "comparison": "Ext2 LateOnly -> Ret LateOnly",
                "sex": sex,
                "metric": "raw_auc_0to3_allPairs",
                "n_cells": int(len(sub)),
                "n_animals": int(sub["animal_id"].nunique()),
                "mean_ext2_late_auc": mean_x,
                "sem_ext2_late_auc": sem_x,
                "mean_ret_late_auc": mean_y,
                "sem_ret_late_auc": sem_y,
                "mean_delta_ret_minus_ext2": mean_delta,
                "paired_t_stat": t_stat,
                "paired_t_p": t_p,
            }
        )

    fig.suptitle("Ext2 LateOnly -> Ret LateOnly\nAll Registered Pairs Raw AUC by Sex", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Ext2_LateOnly_to_Ret_LateOnly_allPairs_rawAUC_bySex.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(OUTDIR / "Ext2_LateOnly_to_Ret_LateOnly_allPairs_rawAUC_values_bySex.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTDIR / "Ext2_LateOnly_to_Ret_LateOnly_allPairs_rawAUC_summary_bySex.csv", index=False)

    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "comparison": "Ext2 LateOnly -> Ret LateOnly",
                "metric": "raw_auc_0to3_allPairs",
                "source_pairs": str(PAIR_FILE),
                "notes": [
                    "Includes all explicit registered Ext2 LateOnly -> Ret LateOnly pairs, not only the strict valid-AUC subset.",
                    "Ext2 raw AUC uses a permissive 0-3 s post-tone recompute when possible and otherwise falls back to the existing raw AUC from the pair table.",
                    "Retrieval raw AUC comes from the explicit pair table.",
                ],
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()
