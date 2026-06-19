#!/usr/bin/env python3
"""Compute per-cell 0-3 s AUC tables for Ext1 and retrieval onset-defined classes."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from freezing_pipeline_all_code_bundle.utils_freezing import infer_fps, load_ext1_npz, load_ret_npz


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
        epochs = tone_epochs_from_id(tone_id)
        if len(epochs) > 1:
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
        epochs.append({"tone_num": order, "tone_order": order, "onset_idx": int(start), "idxs": np.arange(start, stop)})
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
    cell_ids = []
    for idx, col in enumerate(cell_cols):
        norm = normalize_cell_id(col)
        if not norm.startswith("C"):
            norm = f"C{idx:03d}"
        cell_ids.append(norm)
    cell_ids = [cid for cid, keep in zip(cell_ids, valid_mask) if keep]
    if x.shape[1] == 0:
        return None
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


def find_ret_raw_csv(animal: str) -> Path | None:
    raw_dir = ROOT / "ret_raw_trace_files"
    if raw_dir.exists():
        matches = sorted(raw_dir.glob(f"*{animal}*csv"))
        if matches:
            return matches[0]
    zip_path = ROOT / "ret_raw_trace_files 2.zip"
    if not zip_path.exists():
        return None
    tmp_dir = ROOT / "_tmp_ret_auc_raw"
    marker = tmp_dir / ".source_zip"
    previous = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if previous != str(zip_path.resolve()):
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        marker.write_text(str(zip_path.resolve()), encoding="utf-8")
    matches = sorted(tmp_dir.glob(f"**/*{animal}*csv"))
    return matches[0] if matches else None


def best_ret_class_long_path() -> Path:
    candidates = [
        ROOT / "ret_classes_proportions_only/Ret_cellClassifications_long.csv",
        ROOT / "retrieval_pipeline_outputs_bundle/ret_classes_proportions_only 2/Ret_cellClassifications_long.csv",
        ROOT / "retrieval_pipeline_outputs_bundle/ret_classes_proportions_only/Ret_cellClassifications.csv",
    ]
    best_path = None
    best_score = (-1, -1)
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "animal_id" not in df.columns or "cell_id" not in df.columns:
            continue
        score = (df["animal_id"].nunique(), len(df))
        if score > best_score:
            best_path = path
            best_score = score
    if best_path is None:
        raise FileNotFoundError("No retrieval class long CSV found")
    return best_path


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


def compute_auc_from_trace(trace: np.ndarray, tvec: np.ndarray) -> dict[str, float]:
    pre_mask = (tvec >= -5) & (tvec < 0)
    post_mask = (tvec >= 0) & (tvec <= 3)
    pre_vals = trace[pre_mask]
    post_vals = trace[post_mask]
    post_t = tvec[post_mask]
    out = {
        "pre_mean_-5to0": np.nan,
        "auc_0to3_raw": np.nan,
        "auc_0to3_baselineCorrected": np.nan,
        "valid_auc_test": False,
    }
    if np.isfinite(pre_vals).sum() == 0 or np.isfinite(post_vals).sum() < 2:
        return out
    pre_mean = float(np.nanmean(pre_vals))
    valid = np.isfinite(post_t) & np.isfinite(post_vals)
    if valid.sum() < 2:
        return out
    raw_auc = float(np.trapezoid(post_vals[valid], post_t[valid]))
    bc_auc = float(np.trapezoid(post_vals[valid] - pre_mean, post_t[valid]))
    out.update(
        {
            "pre_mean_-5to0": pre_mean,
            "auc_0to3_raw": raw_auc,
            "auc_0to3_baselineCorrected": bc_auc,
            "valid_auc_test": True,
        }
    )
    return out


def build_class_map(long_csv: Path) -> dict[str, dict[str, str]]:
    df = pd.read_csv(long_csv).copy()
    df["animal_id"] = df["animal_id"].map(normalize_animal_id)
    df["cell_id"] = df["cell_id"].map(normalize_cell_id)
    out: dict[str, dict[str, str]] = {}
    for animal, sub in df.groupby("animal_id"):
        out[animal] = dict(zip(sub["cell_id"], sub["class"]))
    return out


def summarize_per_animal(df: pd.DataFrame, session_label: str, class_label: str, pref_col: str, nonpref_col: str):
    valid = df[df["valid_auc_test"]].copy()
    if valid.empty:
        return pd.DataFrame()
    out = (
        valid.groupby("animal_id", as_index=False)
        .agg(
            n_cells=("cell_id", "size"),
            preferred_mean=(pref_col, "mean"),
            preferred_median=(pref_col, "median"),
            nonpreferred_mean=(nonpref_col, "mean"),
            nonpreferred_median=(nonpref_col, "median"),
        )
    )
    out["preferred_minus_nonpreferred_mean"] = out["preferred_mean"] - out["nonpreferred_mean"]
    out["session"] = session_label
    out["cell_class"] = class_label
    cols = [
        "session",
        "cell_class",
        "animal_id",
        "n_cells",
        "preferred_mean",
        "preferred_median",
        "nonpreferred_mean",
        "nonpreferred_median",
        "preferred_minus_nonpreferred_mean",
    ]
    return out[cols]


def process_ext1() -> dict[str, str]:
    outdir = ROOT / "ext1_per_cell_auc_first3_vs_last3_updatedCriteria_2of3_3s"
    outdir.mkdir(parents=True, exist_ok=True)
    class_map = build_class_map(ROOT / "ext1_pipeline_output_2of3_3s_fullregen/Ext1_cellClassifications_long.csv")
    npz_files = sorted((ROOT / "ext1_pipeline_output_2of3_3s_fullregen").glob("*_ext1_zscored_traces.npz"))

    early_rows = []
    late_rows = []
    for npz_path in npz_files:
        animal = normalize_animal_id(npz_path.name.replace("_ext1_zscored_traces.npz", ""))
        cmap = class_map.get(animal, {})
        if not cmap:
            continue
        d = load_ext1_npz(str(npz_path))
        z = np.asarray(d["z"], dtype=float)
        cell_ids = [normalize_cell_id(c) for c in d["cell_ids"]]
        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        pre_f = int(round(5.0 * fps))
        post_f = int(round(5.0 * fps))
        tvec = np.arange(-pre_f, post_f + 1) / fps
        epochs = tone_epochs_from_id(d["tone_id"])
        if len(epochs) < 12:
            continue
        first_onsets = [e["onset_idx"] for e in epochs[:3]]
        last_onsets = [e["onset_idx"] for e in epochs[-3:]]

        for idx, cell_id in enumerate(cell_ids):
            cls = cmap.get(cell_id)
            if cls not in {"EarlyOnly", "LateOnly"}:
                continue
            first_trace = extract_mean_trace(z, first_onsets, idx, pre_f, post_f)
            last_trace = extract_mean_trace(z, last_onsets, idx, pre_f, post_f)
            first_stats = compute_auc_from_trace(first_trace, tvec) if first_trace is not None else {
                "pre_mean_-5to0": np.nan,
                "auc_0to3_raw": np.nan,
                "auc_0to3_baselineCorrected": np.nan,
                "valid_auc_test": False,
            }
            last_stats = compute_auc_from_trace(last_trace, tvec) if last_trace is not None else {
                "pre_mean_-5to0": np.nan,
                "auc_0to3_raw": np.nan,
                "auc_0to3_baselineCorrected": np.nan,
                "valid_auc_test": False,
            }
            row = {
                "animal_id": animal,
                "cell_id": cell_id,
                "valid_auc_test": bool(first_stats["valid_auc_test"] and last_stats["valid_auc_test"]),
                "first_pre_mean_-5to0": first_stats["pre_mean_-5to0"],
                "last_pre_mean_-5to0": last_stats["pre_mean_-5to0"],
                "first_auc_0to3_raw": first_stats["auc_0to3_raw"],
                "last_auc_0to3_raw": last_stats["auc_0to3_raw"],
                "first_auc_0to3_baselineCorrected": first_stats["auc_0to3_baselineCorrected"],
                "last_auc_0to3_baselineCorrected": last_stats["auc_0to3_baselineCorrected"],
                "aucDiff_raw_first_minus_last": first_stats["auc_0to3_raw"] - last_stats["auc_0to3_raw"],
                "aucDiff_bc_first_minus_last": first_stats["auc_0to3_baselineCorrected"]
                - last_stats["auc_0to3_baselineCorrected"],
            }
            if cls == "EarlyOnly":
                early_rows.append(row)
            else:
                late_rows.append(row)

    early_df = pd.DataFrame(early_rows).sort_values(["animal_id", "cell_id"])
    late_df = pd.DataFrame(late_rows).sort_values(["animal_id", "cell_id"])
    early_path = outdir / "Ext1_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    late_path = outdir / "Ext1_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    early_df.to_csv(early_path, index=False)
    late_df.to_csv(late_path, index=False)

    summary = pd.concat(
        [
            summarize_per_animal(
                early_df,
                "Ext1",
                "EarlyOnly",
                "first_auc_0to3_baselineCorrected",
                "last_auc_0to3_baselineCorrected",
            ),
            summarize_per_animal(
                late_df,
                "Ext1",
                "LateOnly",
                "last_auc_0to3_baselineCorrected",
                "first_auc_0to3_baselineCorrected",
            ),
        ],
        ignore_index=True,
    )
    summary_path = outdir / "Ext1_per_cell_AUC_per_animal_summary_updatedCriteria_2of3_3s.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "session": "Ext1",
        "criteria": {
            "thr_z": 0.5,
            "consec_frames": 10,
            "window_s": 3.0,
            "min_hits_per_period": 2,
            "early_tones": [1, 2, 3],
            "late_tones": [10, 11, 12],
        },
        "inputs": {
            "class_long": str(ROOT / "ext1_pipeline_output_2of3_3s_fullregen/Ext1_cellClassifications_long.csv"),
            "npz_dir": str(ROOT / "ext1_pipeline_output_2of3_3s_fullregen"),
        },
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return {
        "early_per_cell": str(early_path),
        "late_per_cell": str(late_path),
        "per_animal_summary": str(summary_path),
    }


def process_ret() -> dict[str, str]:
    outdir = ROOT / "ret_per_cell_auc_tone1_vs_tone4_updatedCriteria_2of3_3s"
    outdir.mkdir(parents=True, exist_ok=True)
    class_long_path = best_ret_class_long_path()
    class_map = build_class_map(class_long_path)
    npz_files = sorted((ROOT / "retrieval_pipeline_outputs_bundle/ret_zscored_npz_bundle").glob("*_retrieval_zscored_traces.npz"))
    session_data = {
        normalize_animal_id(npz_path.name.replace("_retrieval_zscored_traces.npz", "")): load_ret_npz(str(npz_path))
        for npz_path in npz_files
    }
    for animal in sorted(class_map):
        if animal in session_data:
            continue
        raw_csv = find_ret_raw_csv(animal.replace("animal", ""))
        if raw_csv is None:
            continue
        loaded = load_ret_from_raw_csv(raw_csv)
        if loaded is not None:
            session_data[animal] = loaded

    early_rows = []
    late_rows = []
    for animal in sorted(session_data):
        cmap = class_map.get(animal, {})
        if not cmap:
            continue
        d = session_data[animal]
        z = np.asarray(d["z"], dtype=float)
        cell_ids = [normalize_cell_id(c) for c in d["cell_ids"]]
        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        pre_f = int(round(5.0 * fps))
        post_f = int(round(5.0 * fps))
        tvec = np.arange(-pre_f, post_f + 1) / fps
        epochs = tone_epochs_from_arrays(d.get("tone_id"), d.get("tone_flag"))
        tone_lookup = {e["tone_num"]: e["onset_idx"] for e in epochs}
        if 1 not in tone_lookup or 4 not in tone_lookup:
            continue
        early_onsets = [tone_lookup[1]]
        late_onsets = [tone_lookup[4]]

        for idx, cell_id in enumerate(cell_ids):
            cls = cmap.get(cell_id)
            if cls not in {"EarlyOnly", "LateOnly"}:
                continue
            early_trace = extract_mean_trace(z, early_onsets, idx, pre_f, post_f)
            late_trace = extract_mean_trace(z, late_onsets, idx, pre_f, post_f)
            early_stats = compute_auc_from_trace(early_trace, tvec) if early_trace is not None else {
                "pre_mean_-5to0": np.nan,
                "auc_0to3_raw": np.nan,
                "auc_0to3_baselineCorrected": np.nan,
                "valid_auc_test": False,
            }
            late_stats = compute_auc_from_trace(late_trace, tvec) if late_trace is not None else {
                "pre_mean_-5to0": np.nan,
                "auc_0to3_raw": np.nan,
                "auc_0to3_baselineCorrected": np.nan,
                "valid_auc_test": False,
            }
            row = {
                "animal_id": animal,
                "cell_id": cell_id,
                "valid_auc_test": bool(early_stats["valid_auc_test"] and late_stats["valid_auc_test"]),
                "tone1_pre_mean_-5to0": early_stats["pre_mean_-5to0"],
                "tone4_pre_mean_-5to0": late_stats["pre_mean_-5to0"],
                "tone1_auc_0to3_raw": early_stats["auc_0to3_raw"],
                "tone4_auc_0to3_raw": late_stats["auc_0to3_raw"],
                "tone1_auc_0to3_baselineCorrected": early_stats["auc_0to3_baselineCorrected"],
                "tone4_auc_0to3_baselineCorrected": late_stats["auc_0to3_baselineCorrected"],
                "aucDiff_raw_tone1_minus_tone4": early_stats["auc_0to3_raw"] - late_stats["auc_0to3_raw"],
                "aucDiff_bc_tone1_minus_tone4": early_stats["auc_0to3_baselineCorrected"]
                - late_stats["auc_0to3_baselineCorrected"],
            }
            if cls == "EarlyOnly":
                early_rows.append(row)
            else:
                late_rows.append(row)

    early_df = pd.DataFrame(early_rows).sort_values(["animal_id", "cell_id"])
    late_df = pd.DataFrame(late_rows).sort_values(["animal_id", "cell_id"])
    early_path = outdir / "Ret_EarlyOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    late_path = outdir / "Ret_LateOnly_AUC_per_cell_updatedCriteria_2of3_3s.csv"
    early_df.to_csv(early_path, index=False)
    late_df.to_csv(late_path, index=False)

    summary = pd.concat(
        [
            summarize_per_animal(
                early_df,
                "Ret",
                "EarlyOnly",
                "tone1_auc_0to3_baselineCorrected",
                "tone4_auc_0to3_baselineCorrected",
            ),
            summarize_per_animal(
                late_df,
                "Ret",
                "LateOnly",
                "tone4_auc_0to3_baselineCorrected",
                "tone1_auc_0to3_baselineCorrected",
            ),
        ],
        ignore_index=True,
    )
    summary_path = outdir / "Ret_per_cell_AUC_per_animal_summary_updatedCriteria_2of3_3s.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "session": "Ret",
        "criteria": {
            "thr_z": 0.5,
            "consec_frames": 10,
            "window_s": 3.0,
            "early_tone": 1,
            "late_tone": 4,
        },
        "inputs": {
            "class_long": str(class_long_path),
            "npz_dir": str(ROOT / "retrieval_pipeline_outputs_bundle/ret_zscored_npz_bundle"),
            "raw_dir_fallback": str(ROOT / "ret_raw_trace_files"),
        },
    }
    with open(outdir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return {
        "early_per_cell": str(early_path),
        "late_per_cell": str(late_path),
        "per_animal_summary": str(summary_path),
    }


def main() -> None:
    outputs = {"ext1": process_ext1(), "ret": process_ret()}
    with open(ROOT / "per_cell_auc_ext1_ret_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(outputs, fh, indent=2)


if __name__ == "__main__":
    main()
