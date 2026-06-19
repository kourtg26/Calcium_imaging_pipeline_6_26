#!/usr/bin/env python3
"""Compute early/late per-cell AUC tables for all cells in Ext1, Ext2, and retrieval."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from freezing_pipeline_all_code_bundle.utils_freezing import (
    choose_one_ext2_csv_per_animal,
    extract_csvs_from_zip,
    infer_fps,
    load_ext1_npz,
    load_ext2_session_csv,
    load_ret_npz,
)


ROOT = Path(__file__).resolve().parent


def normalize_animal_id(value: object) -> str:
    s = str(value).strip()
    if s.isdigit() and len(s) < 5:
        return f"animal{int(s)}"
    if s.lower() == "cell":
        return "10492"
    return s


def normalize_cell_id(value: object) -> str:
    s = str(value).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"C{int(digits):03d}" if digits else s


def tone_epochs_from_id(tone_id: np.ndarray) -> list[dict]:
    tid = pd.to_numeric(pd.Series(tone_id), errors="coerce").to_numpy()
    uniq = [int(x) for x in np.unique(tid[np.isfinite(tid)]) if int(x) != 0]
    epochs = []
    for tone_num in sorted(uniq):
        idxs = np.where(tid == tone_num)[0]
        if len(idxs) == 0:
            continue
        epochs.append({"tone_num": tone_num, "onset_idx": int(idxs[0]), "idxs": idxs})
    epochs = sorted(epochs, key=lambda d: d["onset_idx"])
    for order, epoch in enumerate(epochs, start=1):
        epoch["tone_order"] = order
    return epochs


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
            epochs = sorted(epochs, key=lambda d: d["onset_idx"])
            for order, epoch in enumerate(epochs, start=1):
                epoch["tone_order"] = order
            return epochs

    if tone_flag is None:
        return []
    flag = pd.to_numeric(pd.Series(tone_flag), errors="coerce").fillna(0).astype(int).to_numpy()
    rises = np.where((flag[1:] == 1) & (flag[:-1] == 0))[0] + 1
    falls = np.where((flag[1:] == 0) & (flag[:-1] == 1))[0] + 1
    if len(flag) and flag[0] == 1:
        rises = np.r_[0, rises]
    if len(flag) and flag[-1] == 1:
        falls = np.r_[falls, len(flag)]
    epochs = []
    for order, (start, stop) in enumerate(zip(rises, falls), start=1):
        epochs.append(
            {
                "tone_num": order,
                "tone_order": order,
                "onset_idx": int(start),
                "idxs": np.arange(start, stop),
            }
        )
    return epochs


def zscore_matrix(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd == 0, np.nan, sd)
    return ((x - mu) / sd).astype(np.float32)


def detect_ret_columns(df: pd.DataFrame):
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    low = {c: c.lower() for c in cols}
    time_col = next((c for c in cols if "time" in low[c]), None)
    tone_id = next((c for c in cols if low[c] in {"cs", "tone_id", "toneid", "cs_id", "toneindex", "tone_index"}), None)
    tone_flag = next(
        (
            c
            for c in cols
            if ("tone" in low[c] or "cs" in low[c])
            and any(k in low[c] for k in ("flag", "within", "is_", "in_", "intone", "in_tone", "toneon", "tone_on"))
        ),
        None,
    )
    freeze_col = next((c for c in cols if "freez" in low[c]), None)
    meta = {c for c in [time_col, tone_id, tone_flag, freeze_col] if c is not None}
    name_cells = [c for c in cols if c not in meta and re.match(r"^C\d+", c, flags=re.IGNORECASE)]
    extra_cells = [c for c in cols if c not in meta and (re.match(r"^\d+$", c.strip()) or low[c].startswith("undecided"))]
    cell_cols = name_cells + [c for c in extra_cells if c not in name_cells]
    if len(cell_cols) < 5:
        cell_cols = [c for c in cols if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    return time_col, tone_id, tone_flag, freeze_col, cell_cols


def load_ret_from_raw_csv(csv_path: Path) -> dict | None:
    df = pd.read_csv(csv_path)
    time_col, tone_id_col, tone_flag_col, freeze_col, cell_cols = detect_ret_columns(df)
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
    cell_ids = []
    for idx, col in enumerate(cell_cols):
        norm = normalize_cell_id(col)
        if not norm.startswith("C"):
            norm = f"C{idx:03d}"
        cell_ids.append(norm)
    cell_ids = [cid for cid, keep in zip(cell_ids, valid_mask) if keep]
    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) if time_col else np.arange(len(df), dtype=float)
    tone_flag = pd.to_numeric(df[tone_flag_col], errors="coerce").fillna(0).astype(int).to_numpy() if tone_flag_col else np.zeros(len(df), dtype=int)
    tone_id = pd.to_numeric(df[tone_id_col], errors="coerce").fillna(0).astype(int).to_numpy() if tone_id_col else tone_flag.copy()
    freeze = pd.to_numeric(df[freeze_col], errors="coerce").fillna(0).astype(int).to_numpy() if freeze_col else np.zeros(len(df), dtype=int)
    return {
        "time": time,
        "z": zscore_matrix(x).astype(float),
        "tone_flag": tone_flag.astype(int),
        "freeze": freeze.astype(int),
        "tone_id": tone_id.astype(int),
        "cell_ids": np.array(cell_ids, dtype=str),
    }


def iter_ret_session_data() -> dict[str, dict]:
    seen = {}
    for npz_path in sorted((ROOT / "retrieval_pipeline_outputs_bundle/ret_zscored_npz_bundle").glob("*_retrieval_zscored_traces.npz")):
        animal = normalize_animal_id(npz_path.name.replace("_retrieval_zscored_traces.npz", ""))
        seen[animal] = load_ret_npz(str(npz_path))

    raw_dir = ROOT / "ret_raw_trace_files"
    csv_paths = sorted(raw_dir.glob("*_ret_*csv")) if raw_dir.exists() else []
    if not csv_paths and (ROOT / "ret_raw_trace_files 2.zip").exists():
        tmp_dir = ROOT / "_tmp_ret_allcells_auc"
        marker = tmp_dir / ".source_zip"
        zip_path = ROOT / "ret_raw_trace_files 2.zip"
        previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if previous != str(zip_path.resolve()):
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp_dir)
            marker.write_text(str(zip_path.resolve()), encoding="utf-8")
        csv_paths = sorted(tmp_dir.glob("**/*_ret_*csv"))

    for csv_path in csv_paths:
        animal = normalize_animal_id(csv_path.name.split("_")[0])
        if animal in seen:
            continue
        loaded = load_ret_from_raw_csv(csv_path)
        if loaded is not None:
            seen[animal] = loaded
    return seen


def extract_mean_trace(z: np.ndarray, onsets: list[int], cell_idx: int, pre_f: int, post_f: int):
    traces = []
    for onset in onsets:
        if onset - pre_f < 0 or onset + post_f >= z.shape[0]:
            continue
        seg = z[onset - pre_f : onset + post_f + 1, cell_idx].astype(float)
        traces.append(seg - z[onset, cell_idx])
    if not traces:
        return None
    return np.nanmean(np.vstack(traces), axis=0)


def compute_auc(trace: np.ndarray, tvec: np.ndarray) -> tuple[float, float]:
    pre_mask = (tvec >= -5) & (tvec < 0)
    post_mask = (tvec >= 0) & (tvec <= 3)
    pre_vals = trace[pre_mask]
    post_vals = trace[post_mask]
    post_t = tvec[post_mask]
    if np.isfinite(pre_vals).sum() == 0 or np.isfinite(post_vals).sum() < 2:
        return np.nan, np.nan
    pre_mean = float(np.nanmean(pre_vals))
    valid = np.isfinite(post_t) & np.isfinite(post_vals)
    if valid.sum() < 2:
        return np.nan, np.nan
    raw = float(np.trapezoid(post_vals[valid], post_t[valid]))
    bc = float(np.trapezoid(post_vals[valid] - pre_mean, post_t[valid]))
    return raw, bc


def write_ext1() -> Path:
    outdir = ROOT / "ext1_allCells_auc_first3_vs_last3_updatedCriteria_2of3_3s"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for npz_path in sorted((ROOT / "ext1_pipeline_output_2of3_3s_fullregen").glob("*_ext1_zscored_traces.npz")):
        animal = normalize_animal_id(npz_path.name.replace("_ext1_zscored_traces.npz", ""))
        d = load_ext1_npz(str(npz_path))
        z = np.asarray(d["z"], dtype=float)
        cell_ids = [normalize_cell_id(c) for c in d["cell_ids"]]
        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        pre_f = int(round(5.0 * fps))
        post_f = int(round(5.0 * fps))
        tvec = np.arange(-pre_f, post_f + 1) / fps
        epochs = tone_epochs_from_arrays(d.get("tone_id"), d.get("tone_flag"))
        if len(epochs) < 12:
            continue
        first_onsets = [e["onset_idx"] for e in epochs[:3]]
        last_onsets = [e["onset_idx"] for e in epochs[-3:]]
        for idx, cell_id in enumerate(cell_ids):
            first_trace = extract_mean_trace(z, first_onsets, idx, pre_f, post_f)
            last_trace = extract_mean_trace(z, last_onsets, idx, pre_f, post_f)
            f_raw, f_bc = compute_auc(first_trace, tvec) if first_trace is not None else (np.nan, np.nan)
            l_raw, l_bc = compute_auc(last_trace, tvec) if last_trace is not None else (np.nan, np.nan)
            rows.append(
                {
                    "animal_id": animal,
                    "cell_id": cell_id,
                    "valid_auc_test": bool(np.isfinite(f_bc) and np.isfinite(l_bc)),
                    "first_auc_0to3_raw": f_raw,
                    "last_auc_0to3_raw": l_raw,
                    "first_auc_0to3_baselineCorrected": f_bc,
                    "last_auc_0to3_baselineCorrected": l_bc,
                }
            )
    out = pd.DataFrame(rows).sort_values(["animal_id", "cell_id"])
    path = outdir / "Ext1_allCells_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    out.to_csv(path, index=False)
    return path


def write_ext2() -> Path:
    outdir = ROOT / "ext2_allCells_auc_first3_vs_last3_updatedCriteria_2of3_3s"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "_tmp_ext2_allcells_auc"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csvs = extract_csvs_from_zip(str(ROOT / "ext2_raw_trace_files.zip"), str(tmp_dir))
    rows = []
    for fp in choose_one_ext2_csv_per_animal(csvs):
        d = load_ext2_session_csv(fp)
        if d is None:
            continue
        animal = normalize_animal_id(Path(fp).name.split("_")[0])
        z = np.asarray(d["z"], dtype=float)
        cell_ids = [normalize_cell_id(c) for c in d["cell_ids"]]
        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        pre_f = int(round(5.0 * fps))
        post_f = int(round(5.0 * fps))
        tvec = np.arange(-pre_f, post_f + 1) / fps
        epochs = tone_epochs_from_arrays(d.get("tone_id"), d.get("tone_flag"))
        if len(epochs) < 12:
            continue
        first_onsets = [e["onset_idx"] for e in epochs[:3]]
        last_onsets = [e["onset_idx"] for e in epochs[-3:]]
        for idx, cell_id in enumerate(cell_ids):
            first_trace = extract_mean_trace(z, first_onsets, idx, pre_f, post_f)
            last_trace = extract_mean_trace(z, last_onsets, idx, pre_f, post_f)
            f_raw, f_bc = compute_auc(first_trace, tvec) if first_trace is not None else (np.nan, np.nan)
            l_raw, l_bc = compute_auc(last_trace, tvec) if last_trace is not None else (np.nan, np.nan)
            rows.append(
                {
                    "animal_id": animal,
                    "cell_id": cell_id,
                    "valid_auc_test": bool(np.isfinite(f_bc) and np.isfinite(l_bc)),
                    "first_auc_0to3_raw": f_raw,
                    "last_auc_0to3_raw": l_raw,
                    "first_auc_0to3_baselineCorrected": f_bc,
                    "last_auc_0to3_baselineCorrected": l_bc,
                }
            )
    out = pd.DataFrame(rows).sort_values(["animal_id", "cell_id"])
    path = outdir / "Ext2_allCells_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    out.to_csv(path, index=False)
    return path


def write_ret() -> Path:
    outdir = ROOT / "ret_allCells_auc_tone1_vs_tone4_updatedCriteria_2of3_3s"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for animal, d in sorted(iter_ret_session_data().items()):
        z = np.asarray(d["z"], dtype=float)
        cell_ids = [normalize_cell_id(c) for c in d["cell_ids"]]
        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        pre_f = int(round(5.0 * fps))
        post_f = int(round(5.0 * fps))
        tvec = np.arange(-pre_f, post_f + 1) / fps
        epochs = tone_epochs_from_arrays(d.get("tone_id"), d.get("tone_flag"))
        lookup = {e["tone_num"]: e["onset_idx"] for e in epochs}
        if 1 not in lookup or 4 not in lookup:
            continue
        for idx, cell_id in enumerate(cell_ids):
            tone1_trace = extract_mean_trace(z, [lookup[1]], idx, pre_f, post_f)
            tone4_trace = extract_mean_trace(z, [lookup[4]], idx, pre_f, post_f)
            t1_raw, t1_bc = compute_auc(tone1_trace, tvec) if tone1_trace is not None else (np.nan, np.nan)
            t4_raw, t4_bc = compute_auc(tone4_trace, tvec) if tone4_trace is not None else (np.nan, np.nan)
            rows.append(
                {
                    "animal_id": animal,
                    "cell_id": cell_id,
                    "valid_auc_test": bool(np.isfinite(t1_bc) and np.isfinite(t4_bc)),
                    "tone1_auc_0to3_raw": t1_raw,
                    "tone4_auc_0to3_raw": t4_raw,
                    "tone1_auc_0to3_baselineCorrected": t1_bc,
                    "tone4_auc_0to3_baselineCorrected": t4_bc,
                }
            )
    out = pd.DataFrame(rows).sort_values(["animal_id", "cell_id"])
    path = outdir / "Ret_allCells_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    outputs = {
        "ext1": str(write_ext1()),
        "ext2": str(write_ext2()),
        "ret": str(write_ret()),
    }
    with open(ROOT / "all_cells_session_auc_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(outputs, fh, indent=2)


if __name__ == "__main__":
    main()
