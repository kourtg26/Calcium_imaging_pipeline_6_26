#!/usr/bin/env python3
"""
Build mean activity traces for classified cells around fixed session anchors.

For each session and group:
  - Use the exact source cell-order CSV already used for the current heatmaps.
  - Keep only cells classified as EarlyOnly or LateOnly.
  - Extract raw session z-scored activity around 30 s and 90 s from session start.
  - Plot mean +/- SEM for each anchor over a -5..5 s window.
  - Export raw-Z and delta-from-anchor CSVs/figures, plus matrices and summary JSON.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_session_first120s_toneResponsive_comparisons import (
    build_time_grid,
    interp_trace,
    load_source_cell_order,
    mean_sem,
    parse_raw_heatmap_csv,
    session_specs,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "classified_sessionAnchor_meanTrace_bundle"
ZIP_PATH = ROOT / "classified_sessionAnchor_meanTrace_bundle.zip"
ANCHORS_S = [30.0, 90.0]
WINDOW_S = (-5.0, 5.0)
TARGET_GROUPS = ["EarlyOnly", "LateOnly"]


def collect_raw_files(raw_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(raw_root.rglob("*.csv")):
        files[path.name] = path
    return files


def extract_window_matrix(meta: pd.DataFrame, raw_root: Path, dt: float, anchor_s: float) -> tuple[np.ndarray, np.ndarray, int]:
    target_t = build_time_grid(WINDOW_S[0], WINDOW_S[1], dt)
    raw_files = collect_raw_files(raw_root)
    raw_cache: dict[str, dict | None] = {}
    rows = []
    n_with_any = 0

    for _, row in meta.iterrows():
        source_file = str(row["source_file"]).strip()
        raw_label = str(row["raw_cell_label"]).strip()
        if source_file not in raw_cache:
            path = raw_files.get(source_file)
            raw_cache[source_file] = parse_raw_heatmap_csv(path) if path is not None else None
        rec = raw_cache[source_file]

        win = np.full(target_t.shape, np.nan, dtype=float)
        if rec is not None and raw_label in rec["raw_to_idx"]:
            idx = int(rec["raw_to_idx"][raw_label])
            time = np.asarray(rec["time"], dtype=float)
            if np.isfinite(time).any():
                time_rel = time - float(time[np.isfinite(time)][0])
                win = interp_trace(time_rel, np.asarray(rec["z"][:, idx], dtype=float), anchor_s + target_t)
                if np.isfinite(win).any():
                    n_with_any += 1
        rows.append(win)

    return np.asarray(rows, dtype=float), target_t, n_with_any


def delta_from_anchor(mat: np.ndarray, t_rel: np.ndarray) -> np.ndarray:
    zero_idx = int(np.argmin(np.abs(t_rel - 0.0)))
    baseline = mat[:, zero_idx:zero_idx + 1]
    return mat - baseline


def plot_anchor_means(
    session_label: str,
    group_label: str,
    anchor_data: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]],
    n_cells_total: int,
    out_png: Path,
    value_label: str,
    title_suffix: str,
) -> pd.DataFrame:
    fig, axes = plt.subplots(1, len(anchor_data), figsize=(6.6 * len(anchor_data), 4.8), gridspec_kw={"wspace": 0.28})
    if len(anchor_data) == 1:
        axes = [axes]

    rows = []
    colors = {30.0: ("#b23b2b", "#e7a18a"), 90.0: ("#1c5e9c", "#9ec0e9")}

    for ax, (anchor_s, t_rel, mean, sem) in zip(axes, anchor_data):
        line_color, fill_color = colors.get(anchor_s, ("#444444", "#c9c9c9"))
        ax.plot(t_rel, mean, color=line_color, linewidth=2.2)
        ax.fill_between(t_rel, mean - sem, mean + sem, color=fill_color, alpha=0.35)
        ax.axvline(0, color="k", linewidth=1.0, alpha=0.45)
        ax.axhline(0, color="k", linewidth=0.9, alpha=0.22)
        ax.set_title(f"{session_label} {group_label}\n{int(anchor_s)} s Session Anchor {title_suffix}")
        ax.set_xlabel("Time From Anchor (s)")
        ax.set_ylabel(value_label)
        ax.set_xlim(WINDOW_S[0], WINDOW_S[1])
        for rel_t, mu, se in zip(t_rel, mean, sem):
            rows.append(
                {
                    "session": session_label,
                    "group": group_label,
                    "anchor_s": float(anchor_s),
                    "time_from_anchor_s": float(rel_t),
                    "mean_z": float(mu) if np.isfinite(mu) else np.nan,
                    "sem_z": float(se) if np.isfinite(se) else np.nan,
                    "n_cells_total": int(n_cells_total),
                }
            )

    fig.suptitle(f"{session_label}: {group_label} cells around fixed session anchors {title_suffix}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def run_one_group(spec: dict, group_label: str) -> dict:
    meta = load_source_cell_order(spec["source_csv"], spec["key"])
    meta = meta[meta["group"] == group_label].copy().reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"No {group_label} cells found for {spec['label']}")

    dt = float(spec["global_dt_s"])
    anchor_results = []
    anchor_results_delta = []
    matrices = {}
    delta_matrices = {}
    counts = {}
    for anchor_s in ANCHORS_S:
        mat, t_rel, n_with_any = extract_window_matrix(meta, Path(spec["raw_root"]), dt, anchor_s)
        delta_mat = delta_from_anchor(mat, t_rel)
        mean, sem = mean_sem(mat)
        delta_mean, delta_sem = mean_sem(delta_mat)
        anchor_results.append((anchor_s, t_rel, mean, sem))
        anchor_results_delta.append((anchor_s, t_rel, delta_mean, delta_sem))
        matrices[f"anchor_{int(anchor_s)}s"] = mat.astype(np.float32)
        delta_matrices[f"anchor_{int(anchor_s)}s"] = delta_mat.astype(np.float32)
        counts[f"anchor_{int(anchor_s)}s_cells_with_any_data"] = int(n_with_any)

    base = OUT_DIR / f"{spec['label']}_{group_label}_sessionAnchor_meanTrace"
    fig_png = base.with_name(base.name + "_30s_90s.png")
    fig_png_delta = base.with_name(base.name + "_30s_90s_deltaFromAnchor.png")
    csv_path = base.with_name(base.name + "_30s_90s.csv")
    csv_path_delta = base.with_name(base.name + "_30s_90s_deltaFromAnchor.csv")
    npz_path = base.with_name(base.name + "_30s_90s_matrices.npz")
    cell_csv = base.with_name(base.name + "_cell_list.csv")
    summary_json = base.with_name(base.name + "_summary.json")

    trace_df = plot_anchor_means(
        spec["label"], group_label, anchor_results, len(meta), fig_png, "Mean Z +/- SEM", "(Raw Z)"
    )
    trace_df_delta = plot_anchor_means(
        spec["label"], group_label, anchor_results_delta, len(meta), fig_png_delta, "Mean Delta Z +/- SEM", "(Delta From Anchor)"
    )
    trace_df.to_csv(csv_path, index=False)
    trace_df_delta.to_csv(csv_path_delta, index=False)
    meta.to_csv(cell_csv, index=False)
    np.savez_compressed(
        npz_path,
        time_from_anchor_s=anchor_results[0][1].astype(np.float32),
        anchor_30s=matrices["anchor_30s"],
        anchor_90s=matrices["anchor_90s"],
        anchor_30s_delta=delta_matrices["anchor_30s"],
        anchor_90s_delta=delta_matrices["anchor_90s"],
    )

    summary = {
        "session": spec["label"],
        "selection_source": str(spec["source_csv"]),
        "classification_group": group_label,
        "n_cells_total": int(len(meta)),
        "anchors_s": ANCHORS_S,
        "window_s": list(WINDOW_S),
        "global_dt_s": dt,
        "counts": counts,
        "delta_metric": "per-cell z-score minus z-score at anchor t=0",
        "source_raw_root": str(spec["raw_root"]),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "files": [fig_png, fig_png_delta, csv_path, csv_path_delta, npz_path, cell_csv, summary_json],
    }


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = [run_one_group(spec, group_label) for spec in session_specs() for group_label in TARGET_GROUPS]
    manifest = {"output_dir": str(OUT_DIR), "sessions": [r["summary"] for r in results]}
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
