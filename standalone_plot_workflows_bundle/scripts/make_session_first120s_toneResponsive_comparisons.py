#!/usr/bin/env python3
"""
Build side-by-side heatmaps and mean-trace comparisons for all tone-onset
responsive cells in each behavior session.

For each session:
  - Left heatmap panel: onset-aligned delta trace from tone 1
    (value minus z at tone onset)
    for each session.
  - Right heatmap panel: first 120 s of the full session.
  - Matching mean-trace comparison plot and CSVs.

Responsive cells are the existing classified cells with class in:
  EarlyOnly / Overlap / LateOnly

Row order:
  - Group order: EarlyOnly, Overlap, LateOnly
  - Within each group: sorted by mean tone 1 delta Z in 0..3 s descending.

The script intentionally reuses the existing class long CSVs so the cell
selection matches prior classification outputs, while fixing retrieval animal-id
normalization and excluding the duplicate "ret" file.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from freezing_pipeline_all_code_bundle.utils_freezing import (
    choose_one_ext2_csv_per_animal,
    extract_csvs_from_zip,
    infer_fps,
    load_ext1_npz,
    load_ext2_session_csv,
    standardize_class_label,
)
from heatmap_pipeline_cli_config_bundle.make_heatmap import (
    find_freeze_col,
    frames_from_seconds,
    find_time_col,
    find_tone_col,
    select_trace_cols,
    tone_bouts_from_flag,
    zscore_cols,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "session_first120s_vs_firstToneSet_toneResponsive_bundle"
ZIP_PATH = ROOT / "session_first120s_vs_firstToneSet_toneResponsive_bundle.zip"
TMP_EXT2 = ROOT / "_tmp_first120_ext2_raw"
TMP_RET = ROOT / "_tmp_first120_ret_raw"

RESPONSIVE_CLASSES = {"EarlyOnly", "Overlap", "LateOnly"}
GROUP_ORDER = ["EarlyOnly", "Overlap", "LateOnly"]
GROUP_COLORS = {
    "EarlyOnly": "#c62828",
    "Overlap": "#7a7a7a",
    "LateOnly": "#1565c0",
}
SESSION_WINDOW_S = 120.0
TONE_WINDOW = (0.0, 5.0)
CLIP_RANGE = (-2.0, 2.0)


def normalize_class_animal_id(value: str, session_key: str) -> str | None:
    s = str(value).strip()
    low = s.lower()
    if not s:
        return None
    if low == "cell":
        return "10492"
    if session_key == "ret" and low == "ret":
        return None
    if low.startswith("animal"):
        digits = "".join(ch for ch in low if ch.isdigit())
        return f"animal{int(digits)}" if digits else s
    if low.isdigit():
        if session_key == "ret" and low in {"1", "2", "3", "4", "5", "6", "8", "9", "10"}:
            return f"animal{int(low)}"
        return str(int(low))
    return s


def load_class_long(csv_path: Path, session_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["animal_id"] = df["animal_id"].map(lambda x: normalize_class_animal_id(x, session_key))
    df = df[df["animal_id"].notna()].copy()
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df["class"] = df["class"].map(standardize_class_label)
    df = df[df["class"].isin(RESPONSIVE_CLASSES)].copy()
    df = df.drop_duplicates(subset=["animal_id", "cell_id", "class"]).reset_index(drop=True)
    return df


def load_group_whitelist(csv_path: Path, session_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("animal_id", "cell_id", "group"):
        if col not in df.columns:
            raise ValueError(f"Whitelist missing required column: {col}")
    df["animal_id"] = df["animal_id"].map(lambda x: normalize_class_animal_id(x, session_key))
    df = df[df["animal_id"].notna()].copy()
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df["group"] = df["group"].map(standardize_class_label)
    df = df[df["group"].isin(RESPONSIVE_CLASSES)].copy()
    df = df.drop_duplicates(subset=["animal_id", "cell_id", "group"]).reset_index(drop=True)
    return df[["animal_id", "cell_id", "group"]]


def load_source_cell_order(csv_path: Path, session_key: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("animal_id", "cell_id", "group", "source_file"):
        if col not in df.columns:
            raise ValueError(f"Source cell-order CSV missing required column: {col}")
    if "raw_cell_label" not in df.columns:
        raise ValueError(f"Source cell-order CSV missing required column: raw_cell_label")
    df["animal_id"] = df["animal_id"].map(lambda x: normalize_class_animal_id(x, session_key))
    df = df[df["animal_id"].notna()].copy()
    df["cell_id"] = df["cell_id"].astype(str).str.strip()
    df["raw_cell_label"] = df["raw_cell_label"].astype(str).str.strip()
    df["source_file"] = df["source_file"].astype(str).map(lambda x: os.path.basename(x).strip())
    df["group"] = df["group"].map(standardize_class_label)
    df = df[df["group"].isin(RESPONSIVE_CLASSES)].copy()
    return df.reset_index(drop=True)


def load_original_heatmap_source(npz_path: Path, csv_path: Path, session_key: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta = load_source_cell_order(csv_path, session_key)
    mat = np.load(npz_path)
    heatmap = np.asarray(mat["heatmap"], dtype=float)
    boundaries = np.asarray(mat["boundaries"], dtype=int)
    if heatmap.shape[0] != len(meta):
        raise ValueError(f"Row mismatch between {csv_path.name} and {npz_path.name}: {len(meta)} vs {heatmap.shape[0]}")
    return meta, heatmap, boundaries


def extract_zip_if_needed(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".source_zip"
    current = str(zip_path.resolve())
    if out_dir.exists() and marker.exists() and marker.read_text(encoding="utf-8").strip() == current:
        if list(out_dir.rglob("*.csv")):
            return
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text(current, encoding="utf-8")


def ext1_npz_paths() -> list[Path]:
    files = sorted((ROOT / "ext1_pipeline_output_2of3_3s_fullregen").glob("*_ext1_zscored_traces.npz"))
    if files:
        return files
    raise FileNotFoundError("No Ext1 NPZ files found in ext1_pipeline_output_2of3_3s_fullregen")


def parse_raw_heatmap_csv(path: Path) -> dict | None:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = find_time_col(df)
    tone_col = find_tone_col(df)
    if time_col is None or tone_col is None:
        return None
    freeze_col = find_freeze_col(df)
    trace_cols = select_trace_cols(df, time_col, tone_col, freeze_col)
    if not trace_cols:
        return None
    time = pd.to_numeric(df[time_col], errors="coerce").values.astype(float)
    tone = pd.to_numeric(df[tone_col], errors="coerce").fillna(0).values.astype(int)
    X = df[trace_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    Z = zscore_cols(X).astype(float)
    tt = time[np.isfinite(time)]
    if tt.size < 3:
        return None
    dt = float(np.median(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    raw_labels = [str(c).strip() for c in trace_cols]
    return {
        "time": time,
        "dt": dt,
        "tone": tone,
        "bouts": tone_bouts_from_flag(tone),
        "z": Z,
        "raw_to_idx": {lab: i for i, lab in enumerate(raw_labels)},
    }


def normalize_ext2_file_animal_id(path: Path) -> str | None:
    base = path.name
    m = re.match(r"(.+?)_ext2_", base)
    aid = m.group(1) if m else base.split("_")[0]
    return normalize_class_animal_id(aid, "ext2")


def ext2_csv_paths() -> list[Path]:
    raw_dir = ROOT / "ext2_raw_trace_files"
    if raw_dir.exists():
        csvs = sorted(raw_dir.rglob("*.csv"))
    else:
        zip_path = ROOT / "ext2_raw_trace_files.zip"
        if not zip_path.exists():
            raise FileNotFoundError("No Ext2 raw CSV directory or ext2_raw_trace_files.zip found")
        csvs = [Path(p) for p in extract_csvs_from_zip(str(zip_path), str(TMP_EXT2))]
    picked = [Path(p) for p in choose_one_ext2_csv_per_animal([str(p) for p in csvs])]
    best: dict[str, tuple[int, Path]] = {}
    for path in picked:
        aid = normalize_ext2_file_animal_id(path)
        if aid is None:
            continue
        size = path.stat().st_size
        if aid not in best or size > best[aid][0]:
            best[aid] = (size, path)
    return [best[k][1] for k in sorted(best)]


def normalize_ret_file_animal_id(path: Path) -> str | None:
    base = path.name.lower()
    if base == "ret_aligned_toneflag.csv":
        return None
    m = re.search(r"animal(\d+)", base)
    if m:
        return f"animal{int(m.group(1))}"
    if "10492" in base or base.startswith("cell_traces_registered_all_days_10492"):
        return "10492"
    m = re.search(r"(10481|10482|10488|10489|10490|10491|11794|11799)", base)
    if m:
        return m.group(1)
    return None


def ret_csv_paths() -> list[Path]:
    zip_path = ROOT / "ret_raw_trace_files 2.zip"
    if not zip_path.exists():
        zip_path = ROOT / "ret_raw_trace_files.zip"
    if not zip_path.exists():
        raise FileNotFoundError("No retrieval raw zip found")
    extract_zip_if_needed(zip_path, TMP_RET)
    best: dict[str, tuple[tuple[int, int], Path]] = {}
    for path in sorted(TMP_RET.rglob("*.csv")):
        if "__macosx" in str(path).lower() or path.name.startswith("._"):
            continue
        aid = normalize_ret_file_animal_id(path)
        if aid is None:
            continue
        explicit = int(aid in path.name)
        score = (explicit, path.stat().st_size)
        if aid not in best or score > best[aid][0]:
            best[aid] = (score, path)
    return [best[k][1] for k in sorted(best)]


def detect_ret_columns(df: pd.DataFrame):
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    lower = {c: c.lower() for c in cols}

    time_col = next((c for c in cols if "time" in lower[c]), None)
    tone_id_col = next(
        (c for c in cols if lower[c] in ("cs", "tone_id", "toneid", "cs_id", "toneindex", "tone_index")),
        None,
    )
    tone_flag_col = next(
        (
            c
            for c in cols
            if ("tone" in lower[c] or "cs" in lower[c])
            and any(k in lower[c] for k in ("flag", "within", "is_", "in_", "intone", "in_tone", "toneon", "tone_on"))
        ),
        None,
    )

    meta = {x for x in (time_col, tone_id_col, tone_flag_col) if x is not None}
    numeric_cols = [c for c in cols if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    name_cells = [c for c in cols if c not in meta and re.match(r"^C\d+", c, flags=re.IGNORECASE)]
    extra = [c for c in cols if c not in meta and (re.match(r"^\d+$", c.strip()) or lower[c].startswith("undecided"))]
    cell_cols = name_cells + [c for c in extra if c not in name_cells]
    if len(cell_cols) < max(5, int(0.4 * len(numeric_cols))):
        cell_cols = numeric_cols
    return time_col, tone_id_col, tone_flag_col, cell_cols


def normalize_ret_cell_ids(cell_cols: list[str], valid_mask: np.ndarray) -> list[str]:
    ids = []
    for idx, c in enumerate(cell_cols):
        digits = re.findall(r"(\d+)", str(c).strip())
        norm = f"C{digits[-1].zfill(3)}" if digits else str(c).strip()
        if not re.match(r"^C\d+", norm, flags=re.IGNORECASE):
            norm = f"C{idx:03d}"
        ids.append(norm)
    return [cid for cid, keep in zip(ids, valid_mask) if keep]


def normalize_ext2_cell_id(cell_id: str) -> str:
    s = str(cell_id).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        return "C" + digits.zfill(3)
    return s


def load_ret_session_csv(path: Path) -> dict | None:
    df = pd.read_csv(path)
    if df.shape[0] < 20 or df.shape[1] < 5:
        return None
    time_col, tone_id_col, tone_flag_col, cell_cols = detect_ret_columns(df)
    if len(cell_cols) < 5:
        return None

    df[cell_cols] = df[cell_cols].apply(
        lambda s: pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    )
    x = df[cell_cols].to_numpy(dtype=float)
    valid_mask = ~np.all(np.isnan(x), axis=0)
    x = x[:, valid_mask]
    if x.shape[1] == 0:
        return None
    cell_ids = normalize_ret_cell_ids(cell_cols, valid_mask)

    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd == 0, np.nan, sd)
    z = ((x - mu) / sd).astype(float)

    if time_col is not None:
        time = pd.to_numeric(df[time_col].values, errors="coerce").astype(float)
    else:
        time = np.arange(len(df), dtype=float)
    if tone_flag_col is not None:
        tone_flag = pd.to_numeric(df[tone_flag_col].values, errors="coerce")
        tone_flag = np.where(np.isfinite(tone_flag), tone_flag, 0.0).astype(int)
    else:
        tone_flag = np.zeros(len(df), dtype=int)
    if tone_id_col is not None:
        tone_id = pd.to_numeric(df[tone_id_col].values, errors="coerce")
        tone_id = np.where(np.isfinite(tone_id), tone_id, 0.0).astype(int)
    else:
        tone_id = np.zeros(len(df), dtype=int)

    return {
        "time": time,
        "z": z,
        "tone_flag": tone_flag,
        "tone_id": tone_id,
        "cell_ids": np.array(cell_ids, dtype=str),
    }


def rising_edges(flag: np.ndarray) -> np.ndarray:
    flag = np.nan_to_num(np.asarray(flag), nan=0.0).astype(int)
    if flag.size == 0:
        return np.array([], dtype=int)
    edges = np.where((flag[1:] == 1) & (flag[:-1] == 0))[0] + 1
    if flag[0] == 1:
        edges = np.r_[0, edges]
    return edges.astype(int)


def find_tone_onsets(tone_flag: np.ndarray, tone_id: np.ndarray) -> list[dict]:
    onsets = rising_edges(tone_flag)
    epochs = []
    for order, onset in enumerate(onsets, start=1):
        tone_num = order
        if onset < len(tone_id):
            val = int(tone_id[onset])
            if val != 0:
                tone_num = val
        epochs.append({"tone_order": order, "tone_num": tone_num, "onset_idx": int(onset)})
    return epochs


def build_time_grid(start: float, stop: float, dt: float) -> np.ndarray:
    n = int(round((stop - start) / dt)) + 1
    return start + np.arange(n, dtype=float) * dt


def interp_trace(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    out = np.full(x_new.shape, np.nan, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return out
    xv = x[valid]
    yv = y[valid]
    order = np.argsort(xv)
    xv = xv[order]
    yv = yv[order]
    if xv[0] == xv[-1]:
        return out
    mask = (x_new >= xv[0]) & (x_new <= xv[-1])
    if mask.any():
        out[mask] = np.interp(x_new[mask], xv, yv)
    return out


def resample_session_matrix(time: np.ndarray, z: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    time_rel = np.asarray(time, dtype=float) - float(np.asarray(time, dtype=float)[0])
    rows = [interp_trace(time_rel, z[:, j], target_t) for j in range(z.shape[1])]
    return np.asarray(rows, dtype=float)


def resample_tone_mean_matrix(time: np.ndarray, z: np.ndarray, onset_indices: list[int], target_t: np.ndarray) -> np.ndarray:
    rows = []
    time = np.asarray(time, dtype=float)
    for j in range(z.shape[1]):
        per_tone = []
        for onset_idx in onset_indices:
            onset_time = float(time[onset_idx])
            per_tone.append(interp_trace(time, z[:, j], onset_time + target_t))
        rows.append(np.nanmean(np.asarray(per_tone, dtype=float), axis=0) if per_tone else np.full(target_t.shape, np.nan))
    return np.asarray(rows, dtype=float)


def tone_delta_from_onset(mat: np.ndarray, tone_t: np.ndarray) -> np.ndarray:
    onset_idx = int(np.argmin(np.abs(tone_t - 0.0)))
    baseline = mat[:, onset_idx:onset_idx + 1]
    return mat - baseline


def mean_sem(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(np.isfinite(mat), axis=0)
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0, ddof=1)
    sem = np.where(n > 1, sd / np.sqrt(n), 0.0)
    sem = np.where(np.isfinite(mean), sem, np.nan)
    return mean, sem


def xtick_positions(values: np.ndarray, ticks: list[float]) -> tuple[list[float], list[str]]:
    pos = []
    labels = []
    for tick in ticks:
        idx = int(np.argmin(np.abs(values - tick)))
        pos.append(idx)
        labels.append(str(int(tick)) if float(tick).is_integer() else f"{tick:g}")
    return pos, labels


def plot_heatmaps(
    session_label: str,
    tone_t: np.ndarray,
    tone_mat: np.ndarray,
    session_t: np.ndarray,
    session_mat: np.ndarray,
    meta: pd.DataFrame,
    out_png: Path,
) -> None:
    n_cells = tone_mat.shape[0]
    fig_h = min(22, max(6, n_cells / 95))
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(19, fig_h),
        gridspec_kw={"width_ratios": [1.0, 3.2, 8.5, 0.28], "wspace": 0.08},
    )
    ax_group, ax_tone, ax_session, cax = axes
    cmap = plt.get_cmap("viridis")
    vmin, vmax = CLIP_RANGE

    im0 = ax_tone.imshow(np.clip(tone_mat, vmin, vmax), aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax_session.imshow(np.clip(session_mat, vmin, vmax), aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    ax_group.set_xlim(0, 1)
    ax_group.set_ylim(n_cells - 0.5, -0.5)
    ax_group.axis("off")

    tone_ticks, tone_labels = xtick_positions(tone_t, [0, 1, 3, 5])
    sess_ticks, sess_labels = xtick_positions(session_t, [0, 30, 60, 90, 120])
    ax_tone.set_xticks(tone_ticks)
    ax_tone.set_xticklabels(tone_labels)
    ax_session.set_xticks(sess_ticks)
    ax_session.set_xticklabels(sess_labels)
    ax_tone.set_xlabel("Time From Onset (s)")
    ax_session.set_xlabel("Session Time (s)")
    ax_tone.set_title(f"{session_label} Tone 1 Delta From Onset")
    ax_session.set_title(f"{session_label} First 120 s Session")
    ax_tone.set_ylabel("Tone-Responsive Cells")
    ax_session.set_yticks([])

    y0 = 0
    for grp in GROUP_ORDER:
        grp_n = int((meta["group"] == grp).sum())
        if grp_n <= 0:
            continue
        ax_group.add_patch(
            Rectangle((0.42, y0 - 0.5), 0.18, grp_n, facecolor=GROUP_COLORS[grp], edgecolor="k", linewidth=0.6)
        )
        ax_group.text(
            0.38,
            y0 + (grp_n - 1) / 2.0,
            f"{grp}\n(n={grp_n})",
            ha="right",
            va="center",
            fontsize=9,
            color="black",
        )
        y0 += grp_n

    y = 0
    for grp in GROUP_ORDER[:-1]:
        y += int((meta["group"] == grp).sum())
        ax_group.axhline(y - 0.5, color="k", linewidth=0.9, alpha=0.35)
        ax_tone.axhline(y - 0.5, color="k", linewidth=0.9, alpha=0.35)
        ax_session.axhline(y - 0.5, color="k", linewidth=0.9, alpha=0.35)

    cb = fig.colorbar(im0, cax=cax)
    cb.set_label("Z / Delta Z (clipped -1 to 1)")

    group_counts = meta["group"].value_counts().reindex(GROUP_ORDER, fill_value=0).to_dict()
    fig.suptitle(
        f"{session_label}: all tone-onset responsive cells\n"
        f"Group counts {group_counts} | n={n_cells}",
        y=0.995,
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_comparison(
    session_label: str,
    tone_t: np.ndarray,
    tone_mat: np.ndarray,
    session_t: np.ndarray,
    session_mat: np.ndarray,
    out_png: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tone_mean, tone_sem = mean_sem(tone_mat)
    session_mean, session_sem = mean_sem(session_mat)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"wspace": 0.28})
    ax0, ax1 = axes

    ax0.plot(tone_t, tone_mean, color="#9f2f23", linewidth=2.2)
    ax0.fill_between(tone_t, tone_mean - tone_sem, tone_mean + tone_sem, color="#e6a07a", alpha=0.35)
    ax0.axvline(0, color="k", linewidth=1.0, alpha=0.45)
    ax0.axhline(0, color="k", linewidth=0.9, alpha=0.25)
    ax0.set_title(f"{session_label} Tone 1 Delta From Onset")
    ax0.set_xlabel("Time From Onset (s)")
    ax0.set_ylabel("Mean Delta Z +/- SEM")

    ax1.plot(session_t, session_mean, color="#1f5aa6", linewidth=2.2)
    ax1.fill_between(session_t, session_mean - session_sem, session_mean + session_sem, color="#99b7ea", alpha=0.35)
    ax1.axhline(0, color="k", linewidth=0.9, alpha=0.25)
    ax1.set_title(f"{session_label} First 120 s Mean")
    ax1.set_xlabel("Session Time (s)")
    ax1.set_ylabel("Mean Z +/- SEM")

    fig.suptitle(f"{session_label}: cell-weighted mean comparison across all responsive cells", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    tone_df = pd.DataFrame(
        {
            "time_s": tone_t,
            "mean_z": tone_mean,
            "sem_z": tone_sem,
            "n_cells": tone_mat.shape[0],
        }
    )
    session_df = pd.DataFrame(
        {
            "time_s": session_t,
            "mean_z": session_mean,
            "sem_z": session_sem,
            "n_cells": session_mat.shape[0],
        }
    )
    return tone_df, session_df


def session_specs() -> list[dict]:
    return [
        {
            "key": "ext1",
            "label": "Ext1",
            "source_csv": ROOT / "Ext1_fulltone20s_3s2of3_fullregen_toneOnly_heatmap_cell_order.csv",
            "source_npz": ROOT / "Ext1_fulltone20s_3s2of3_fullregen_toneOnly_heatmap_matrix.npz",
            "raw_root": ROOT / "_ext1_fulltone20s_heatmap_work_3s2of3_fullregen",
            "global_dt_s": 0.07992058518067324,
        },
        {
            "key": "ext2",
            "label": "Ext2",
            "source_csv": ROOT / "Ext2_fulltone20s_3s2of3_fullregen_toneOnly_heatmap_cell_order.csv",
            "source_npz": ROOT / "Ext2_fulltone20s_3s2of3_fullregen_toneOnly_heatmap_matrix.npz",
            "raw_root": ROOT / "_ext2_fulltone20s_heatmap_work_3s2of3_fullregen" / "ext2_raw_trace_files",
            "global_dt_s": 0.0908,
        },
        {
            "key": "ret",
            "label": "Retrieval",
            "source_csv": ROOT / "Ret_first5s_toneOnly_heatmap_cell_order.csv",
            "source_npz": ROOT / "Ret_first5s_toneOnly_heatmap_matrix.npz",
            "raw_root": ROOT / "_ret_first5s_heatmap_work_v4" / "ret_raw_trace_files",
            "global_dt_s": 0.0799000000115484,
        },
    ]


def file_animal_id(path: Path, session_key: str) -> str | None:
    if session_key == "ext1":
        base = path.name.replace("_ext1_zscored_traces.npz", "")
        return normalize_class_animal_id(base, session_key)
    if session_key == "ext2":
        return normalize_ext2_file_animal_id(path)
    if session_key == "ret":
        return normalize_ret_file_animal_id(path)
    return None


def run_one_session(spec: dict) -> dict:
    meta, source_heatmap, boundaries = load_original_heatmap_source(spec["source_npz"], spec["source_csv"], spec["key"])
    first_tone_len = int(boundaries[0])
    tone_exact = np.asarray(source_heatmap[:, :first_tone_len], dtype=float)
    dt_target = float(spec["global_dt_s"])
    first_tone_frames = min(first_tone_len, frames_from_seconds(dt_target, TONE_WINDOW[1]))
    tone_t = np.arange(first_tone_frames, dtype=float) * dt_target
    session_t = build_time_grid(0.0, SESSION_WINDOW_S, dt_target)

    raw_files = {p.name: p for p in sorted(Path(spec["raw_root"]).rglob("*.csv"))}
    raw_cache: dict[str, dict | None] = {}

    session_rows = []
    animals_used = []
    animal_counts: dict[str, int] = {}

    for _, row in meta.iterrows():
        animal_id = str(row["animal_id"])
        source_file = str(row["source_file"]).strip()
        raw_label = str(row["raw_cell_label"]).strip()
        if source_file not in raw_cache:
            path = raw_files.get(source_file)
            raw_cache[source_file] = parse_raw_heatmap_csv(path) if path is not None else None
        rec = raw_cache[source_file]

        sess_seg = np.full(session_t.shape, np.nan, dtype=float)
        if rec is not None and rec["bouts"] and raw_label in rec["raw_to_idx"]:
            idx = int(rec["raw_to_idx"][raw_label])
            time_rel = np.asarray(rec["time"], dtype=float) - float(np.asarray(rec["time"], dtype=float)[0])
            sess_seg = interp_trace(time_rel, np.asarray(rec["z"][:, idx], dtype=float), session_t)
            animal_counts[animal_id] = animal_counts.get(animal_id, 0) + 1

        session_rows.append(sess_seg)

    tone_all = np.asarray(tone_exact[:, :first_tone_frames], dtype=float)
    session_all = np.asarray(session_rows, dtype=float)
    meta = meta.copy()
    post_mask = (tone_t >= 0) & (tone_t <= 3)
    meta["sort_score_0to3s"] = np.nanmean(tone_all[:, post_mask], axis=1)
    meta["peak_tone1_0to3s"] = np.nanmax(tone_all[:, post_mask], axis=1)
    meta["peak_late_0to3s"] = np.nan
    meta["mean_tone_0to3s"] = meta["sort_score_0to3s"]
    meta["mean_session_0to120s"] = np.nanmean(session_all, axis=1)
    meta["sort_rank"] = np.arange(len(meta), dtype=int)
    for aid in sorted(set(meta["animal_id"].astype(str))):
        src_files = sorted(set(meta.loc[meta["animal_id"].astype(str) == aid, "source_file"].astype(str)))
        animals_used.append(
            {
                "animal_id": aid,
                "file": ", ".join(src_files),
                "n_responsive_cells": int(animal_counts.get(aid, 0)),
            }
        )

    base = OUT_DIR / f"{spec['label']}_first120s_vs_firstToneSet_toneResponsive"
    heatmap_png = base.with_name(base.name + "_heatmap_viridis_clip-m2to2.png")
    mean_png = base.with_name(base.name + "_meanComparison.png")
    cell_order_csv = base.with_name(base.name + "_cell_order.csv")
    tone_csv = base.with_name(base.name + "_firstToneSet_meanTrace.csv")
    session_csv = base.with_name(base.name + "_first120s_meanTrace.csv")
    matrix_npz = base.with_name(base.name + "_matrices.npz")
    summary_json = base.with_name(base.name + "_summary.json")

    plot_heatmaps(spec["label"], tone_t, tone_all, session_t, session_all, meta, heatmap_png)
    tone_df, session_df = plot_mean_comparison(spec["label"], tone_t, tone_all, session_t, session_all, mean_png)

    meta.drop(columns=["group_rank"], errors="ignore").to_csv(cell_order_csv, index=False)
    tone_df.to_csv(tone_csv, index=False)
    session_df.to_csv(session_csv, index=False)
    np.savez_compressed(
        matrix_npz,
        tone_heatmap=tone_all.astype(np.float32),
        session_heatmap=session_all.astype(np.float32),
        tone_t=tone_t.astype(np.float32),
        session_t=session_t.astype(np.float32),
    )

    summary = {
        "session": spec["label"],
        "classification_source": str(spec["source_csv"]),
        "selection_source": str(spec["source_csv"]),
        "responsive_classes": sorted(RESPONSIVE_CLASSES),
        "group_counts": meta["group"].value_counts().reindex(GROUP_ORDER, fill_value=0).to_dict(),
        "n_cells_total": int(len(meta)),
        "n_animals": int(len(animals_used)),
        "animals_used": animals_used,
        "panel_tones": [1],
        "late_tones": [],
        "session_window_s": SESSION_WINDOW_S,
        "tone_window_s": list(TONE_WINDOW),
        "clip_range": list(CLIP_RANGE),
        "global_dt_s": dt_target,
        "sorting": "Exact row order from source heatmap cell_order CSV",
        "tone_panel_metric": "Exact first-tone values from source heatmap matrix cropped to 0..5 s from tone onset",
        "session_panel_metric": "Raw session z-score over first 120 s using the exact source heatmap cell list",
        "source_matrix": str(spec["source_npz"]),
        "source_raw_root": str(spec["raw_root"]),
    }
    summary["whitelist_counts"] = meta["group"].value_counts().reindex(GROUP_ORDER, fill_value=0).to_dict()
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "session": spec["label"],
        "summary": summary,
        "files": [
            heatmap_png,
            mean_png,
            cell_order_csv,
            tone_csv,
            session_csv,
            matrix_npz,
            summary_json,
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_one_session(spec) for spec in session_specs()]

    manifest = {
        "output_dir": str(OUT_DIR),
        "sessions": [r["summary"] for r in results],
    }
    manifest_path = OUT_DIR / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.iterdir()):
            if path.is_file():
                zf.write(path, arcname=path.name)

    print(f"Wrote {OUT_DIR}")
    print(f"Wrote {ZIP_PATH}")


if __name__ == "__main__":
    main()
