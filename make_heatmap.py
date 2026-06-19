#!/usr/bin/env python3
"""
make_heatmap.py
Build a "tone-only" heatmap (cells x concatenated tone time) with onset-evoked group
classification (EarlyOnly / Overlap / LateOnly / Neither) and group-specific gradients.

Designed to reproduce the Ext2-style heatmap pipeline with parameters moved to a YAML/JSON config.

Example:
  python make_heatmap.py --config ext1_config.yaml
"""

import os, re, glob, zipfile, json, hashlib, argparse, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yaml
except Exception:
    yaml = None

# -----------------------------
# Config loading / validation
# -----------------------------
def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if p.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not available; use JSON config instead.")
        with open(p, "r") as f:
            return yaml.safe_load(f)
    if p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f)
    # try YAML as default
    if yaml is not None:
        with open(p, "r") as f:
            return yaml.safe_load(f)
    raise RuntimeError("Unknown config format. Use .yaml/.yml or .json")

def deep_update(base: dict, override: dict) -> dict:
    """Recursively update dict."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# Parsing helpers
# -----------------------------
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def infer_animal_id(filename: str) -> str:
    base = os.path.basename(filename)
    m = re.match(r"^([A-Za-z]*\d+)", base)
    if m:
        return m.group(1)
    return base.split("_")[0]

def find_time_col(cols):
    exact = ["Time_s", "Time (s)", "Time(s)", "Time"]
    for c in exact:
        if c in cols:
            return c
    for c in cols:
        if "time" in str(c).lower():
            return c
    return None

def _is_binary_series(s: pd.Series, min_valid_frac=0.5, min_bin_frac=0.95) -> bool:
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().mean() < min_valid_frac:
        return False
    vv = v.dropna().values
    return np.isin(vv, [0, 1]).mean() >= min_bin_frac

def find_tone_col(df: pd.DataFrame):
    cols = list(df.columns)
    priority = ["ToneFlag", "WithinTone", "Within_Tone", "InTone", "In_Tone", "is_tone", "CS"]
    for c in priority:
        if c in cols and _is_binary_series(df[c]):
            return c
    for c in cols:
        lc = str(c).lower()
        if ("tone" in lc or "intone" in lc or "cs" == lc) and _is_binary_series(df[c]):
            return c
    return None

def find_freeze_col(df: pd.DataFrame):
    cols = list(df.columns)
    priority = ["FreezeFlag", "Freezing", "freezing", "Freezing_Index", "FreezingIndex"]
    for c in priority:
        if c in cols and _is_binary_series(df[c]):
            return c
    for c in cols:
        if "freez" in str(c).lower() and _is_binary_series(df[c]):
            return c
    return None

def select_trace_cols(df: pd.DataFrame, time_col: str, tone_col: str, freeze_col: str | None):
    cols = list(df.columns)
    exclude = {time_col, tone_col}
    if freeze_col is not None:
        exclude.add(freeze_col)

    def is_aux(c):
        lc = str(c).lower()
        if c in exclude:
            return True
        if "tone" in lc and c != tone_col:
            return True
        if "freez" in lc and freeze_col is not None and c != freeze_col:
            return True
        if "time" in lc and c != time_col:
            return True
        return False

    # Prefer columns like C###
    trace_cols = []
    for c in cols:
        if is_aux(c):
            continue
        name = str(c).strip()
        if re.fullmatch(r"C\d+", name):
            trace_cols.append(c)
    if trace_cols:
        return trace_cols

    # Fallback: numeric columns with near-complete data
    numeric_cols = []
    for c in cols:
        if is_aux(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.95 and np.nanstd(s.values) > 1e-12:
            numeric_cols.append(c)
    return numeric_cols

def tone_bouts_from_flag(flag_01: np.ndarray):
    f = np.asarray(flag_01).astype(int)
    on  = np.where((f[1:] == 1) & (f[:-1] == 0))[0] + 1
    off = np.where((f[1:] == 0) & (f[:-1] == 1))[0] + 1
    if f[0] == 1:
        on = np.r_[0, on]
    if f[-1] == 1:
        off = np.r_[off, len(f)]
    bouts = []
    for s, e in zip(on, off):
        if e > s:
            bouts.append((int(s), int(e)))
    return bouts

def zscore_cols(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd

def has_run_ge(x: np.ndarray, thresh: float, run_len: int) -> bool:
    ok = (x >= thresh).astype(np.int8)
    if ok.sum() < run_len:
        return False
    run = 0
    for v in ok:
        if v:
            run += 1
            if run >= run_len:
                return True
        else:
            run = 0
    return False

def frames_from_seconds(dt: float, seconds: float) -> int:
    if dt <= 0:
        return 1
    return max(1, int(np.floor(seconds / dt)))

# -----------------------------
# Core pipeline
# -----------------------------
def build_heatmap(cfg: dict, dry_run: bool = False) -> dict:
    out_prefix = str(cfg.get("out_prefix", "ExtX")).strip()
    out_dir = ensure_dir(cfg.get("out_dir", "/mnt/data"))
    work_dir = ensure_dir(cfg.get("work_dir", os.path.join(out_dir, f"_{out_prefix.lower()}_work")))

    input_zips = cfg.get("input_zips", [])
    if not input_zips:
        raise ValueError("Config must include input_zips (list of .zip files).")

    # onset-evoked params
    oe = cfg.get("onset_evoked", {}) or {}
    dz_thresh = float(oe.get("dz_thresh", 0.5))
    consec_frames = int(oe.get("consec_frames", 10))
    post_win_s = float(oe.get("post_win_s", 3.0))
    dz_reference = str(oe.get("dz_reference", "onset_sample"))  # onset_sample | pre_window
    pre_win_s = float(oe.get("pre_win_s", 1.0))  # only used if pre_window

    # grouping params
    grp = cfg.get("groups", {}) or {}
    early_n = int(grp.get("early_n", 3))
    late_n  = int(grp.get("late_n", 3))

    # heatmap params
    hm = cfg.get("heatmap", {}) or {}
    tone_only = bool(hm.get("tone_only", True))
    tone_len_policy = str(hm.get("tone_len_policy", "min_across_all"))  # min_across_all | fixed_frames | fixed_seconds
    fixed_tone_len_frames = hm.get("fixed_tone_len_frames", None)
    fixed_tone_len_seconds = hm.get("fixed_tone_len_seconds", None)

    include_groups = hm.get("include_groups_in_heatmap", ["EarlyOnly", "Overlap", "LateOnly"])
    include_groups = [str(g) for g in include_groups]

    clip_min = float(hm.get("clip_min", -1.0))
    clip_max = float(hm.get("clip_max",  1.0))

    # color anchors
    colors = cfg.get("colors", {}) or {}
    def _col(name, default):
        v = colors.get(name, default)
        arr = np.array(v, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError(f"Color for {name} must be 3 values RGB in [0..1].")
        return arr
    base = {
        "EarlyOnly": _col("EarlyOnly", [1.0, 0.0, 0.0]),
        "Overlap":   _col("Overlap",   [0.5, 0.5, 0.5]),
        "LateOnly":  _col("LateOnly",  [0.0, 0.0, 1.0]),
    }
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # plot params
    plot = cfg.get("plot", {}) or {}
    fig_w = float(plot.get("fig_w", 15))
    cells_per_inch = float(plot.get("cells_per_inch", 140))
    fig_h_min = float(plot.get("fig_h_min", 5))
    fig_h_max = float(plot.get("fig_h_max", 18))
    vline_lw = float(plot.get("tone_separator_lw", 0.6))
    vline_alpha = float(plot.get("tone_separator_alpha", 0.25))
    hline_lw = float(plot.get("group_separator_lw", 1.0))
    hline_alpha = float(plot.get("group_separator_alpha", 0.35))
    dpi = int(plot.get("dpi", 300))

    # output naming
    npz_name = f"{out_prefix}_toneOnly_heatmap_matrix.npz"
    csv_name = f"{out_prefix}_toneOnly_heatmap_cell_order.csv"
    png_name = f"{out_prefix}_toneOnly_heatmap_groupSpecificGradients_clip{clip_min:g}to{clip_max:g}_NO_LABELS.png"
    bundle_name = f"{out_prefix.lower()}_toneOnly_heatmap_earlyTop_lateBottom_bundle.zip"
    bundle_png_only_name = f"{out_prefix.lower()}_toneOnly_heatmap_clip{clip_min:g}to{clip_max:g}_NO_LABELS_bundle.zip"
    meta_name = f"{out_prefix}_toneOnly_heatmap_run_metadata.json"

    npz_out = os.path.join(out_dir, npz_name)
    csv_out = os.path.join(out_dir, csv_name)
    png_out = os.path.join(out_dir, png_name)
    bundle_out = os.path.join(out_dir, bundle_name)
    bundle_png_out = os.path.join(out_dir, bundle_png_only_name)
    meta_out = os.path.join(out_dir, meta_name)

    # Extract zips
    if dry_run:
        print(f"[DRY RUN] Would extract {len(input_zips)} zip(s) into {work_dir}")
    else:
        for zp in input_zips:
            zp = os.path.expanduser(str(zp))
            if not os.path.exists(zp):
                raise FileNotFoundError(f"Input zip not found: {zp}")
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(work_dir)

    # Collect CSVs
    csv_paths = sorted(glob.glob(os.path.join(work_dir, "*.csv")))
    if not csv_paths:
        raise RuntimeError(f"No CSVs found in work_dir: {work_dir}")

    # Load per-animal records
    records = []
    excluded = []
    for fp in csv_paths:
        try:
            df = pd.read_csv(fp)
            time_col = find_time_col(df.columns)
            if time_col is None:
                excluded.append((fp, "no_time_col"))
                continue
            tone_col = find_tone_col(df)
            if tone_col is None:
                excluded.append((fp, "no_tone_col"))
                continue
            freeze_col = find_freeze_col(df)
            trace_cols = select_trace_cols(df, time_col, tone_col, freeze_col)
            if len(trace_cols) == 0:
                excluded.append((fp, "no_trace_cols"))
                continue

            t = pd.to_numeric(df[time_col], errors="coerce").values.astype(np.float64)
            tone = pd.to_numeric(df[tone_col], errors="coerce").fillna(0).values.astype(int)
            freeze = None
            if freeze_col is not None:
                freeze = pd.to_numeric(df[freeze_col], errors="coerce").fillna(0).values.astype(int)

            X = df[trace_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

            # clean cell IDs
            cell_ids = []
            for j, c in enumerate(trace_cols):
                name = str(c).strip()
                if re.fullmatch(r"C\d+", name):
                    cell_ids.append(name)
                else:
                    cell_ids.append(f"C{j:03d}")

            tt = t[np.isfinite(t)]
            if len(tt) < 3:
                excluded.append((fp, "bad_time_vector"))
                continue
            dt = float(np.median(np.diff(tt)))
            if not np.isfinite(dt) or dt <= 0:
                excluded.append((fp, "bad_dt"))
                continue

            bouts = tone_bouts_from_flag(tone)
            if len(bouts) < (early_n + late_n):
                excluded.append((fp, f"too_few_tones_{len(bouts)}"))
                continue

            animal_id = infer_animal_id(fp)
            records.append(dict(
                animal_id=animal_id,
                file=fp,
                t=t,
                dt=dt,
                tone=tone,
                freeze=freeze,
                X=X,
                cell_ids=cell_ids,
                bouts=bouts,
            ))
        except Exception as e:
            excluded.append((fp, f"read_error:{type(e).__name__}:{e}"))

    if not records:
        raise RuntimeError("No valid CSVs found (need time + binary tone flag + trace columns).")

    # Harmonize tone count across animals
    n_tones = min(len(r["bouts"]) for r in records)

    # Determine tone length in frames
    if tone_len_policy == "min_across_all":
        tone_len_frames = min((e - s) for r in records for (s, e) in r["bouts"][:n_tones])
    elif tone_len_policy == "fixed_frames":
        if fixed_tone_len_frames is None:
            raise ValueError("fixed_tone_len_frames required when tone_len_policy=fixed_frames.")
        tone_len_frames = int(fixed_tone_len_frames)
    elif tone_len_policy == "fixed_seconds":
        if fixed_tone_len_seconds is None:
            raise ValueError("fixed_tone_len_seconds required when tone_len_policy=fixed_seconds.")
        # use minimum dt across animals for a conservative frame count
        dt_min = min(r["dt"] for r in records)
        tone_len_frames = frames_from_seconds(dt_min, float(fixed_tone_len_seconds))
    else:
        raise ValueError(f"Unknown tone_len_policy: {tone_len_policy}")

    if tone_len_frames <= 0:
        raise RuntimeError("Computed tone_len_frames <= 0")

    # Build meta rows + heatmap rows aligned to meta row order
    meta_rows = []
    heat_rows = []

    for r in records:
        animal = r["animal_id"]
        dt = r["dt"]
        bouts = r["bouts"][:n_tones]
        X = r["X"]
        cell_ids = r["cell_ids"]

        Z = zscore_cols(X)  # (T x Ncells)

        post_frames = frames_from_seconds(dt, post_win_s)
        pre_frames  = frames_from_seconds(dt, pre_win_s)

        early_idx = list(range(min(early_n, n_tones)))
        late_idx  = list(range(max(0, n_tones - late_n), n_tones))

        for j, cid in enumerate(cell_ids):
            early_peak = -np.inf
            late_peak  = -np.inf

            early_hits = []
            late_hits  = []

            for k in early_idx:
                s, _e = bouts[k]
                idx0 = s
                w_end = min(Z.shape[0], idx0 + post_frames + 1)

                if dz_reference == "onset_sample":
                    ref = Z[idx0, j]
                elif dz_reference == "pre_window":
                    w0 = max(0, idx0 - pre_frames)
                    ref = float(np.nanmean(Z[w0:idx0+1, j]))
                else:
                    raise ValueError(f"Unknown dz_reference: {dz_reference}")

                dz = Z[idx0:w_end, j] - ref
                if dz.size:
                    early_peak = max(early_peak, float(np.nanmax(dz)))
                    early_hits.append(has_run_ge(dz, dz_thresh, consec_frames))
                else:
                    early_hits.append(False)

            for k in late_idx:
                s, _e = bouts[k]
                idx0 = s
                w_end = min(Z.shape[0], idx0 + post_frames + 1)

                if dz_reference == "onset_sample":
                    ref = Z[idx0, j]
                elif dz_reference == "pre_window":
                    w0 = max(0, idx0 - pre_frames)
                    ref = float(np.nanmean(Z[w0:idx0+1, j]))
                else:
                    raise ValueError(f"Unknown dz_reference: {dz_reference}")

                dz = Z[idx0:w_end, j] - ref
                if dz.size:
                    late_peak = max(late_peak, float(np.nanmax(dz)))
                    late_hits.append(has_run_ge(dz, dz_thresh, consec_frames))
                else:
                    late_hits.append(False)

            early_resp = bool(np.any(early_hits))
            late_resp  = bool(np.any(late_hits))

            if early_resp and (not late_resp):
                group = "EarlyOnly"
                sort_score = early_peak
            elif late_resp and (not early_resp):
                group = "LateOnly"
                sort_score = late_peak
            elif early_resp and late_resp:
                group = "Overlap"
                sort_score = max(early_peak, late_peak)
            else:
                group = "Neither"
                sort_score = -np.inf

            meta_rows.append(dict(
                animal_id=animal,
                cell_id=cid,
                group=group,
                early_peak_dZ=early_peak if np.isfinite(early_peak) else np.nan,
                late_peak_dZ=late_peak if np.isfinite(late_peak) else np.nan,
                sort_score=sort_score if np.isfinite(sort_score) else np.nan,
                dt=dt,
                source_file=os.path.basename(r["file"]),
            ))

            if tone_only:
                segs = []
                for k in range(n_tones):
                    s, e = bouts[k]
                    seg = Z[s:s + tone_len_frames, j]
                    # pad/truncate to tone_len_frames
                    if seg.shape[0] != tone_len_frames:
                        pad = tone_len_frames - seg.shape[0]
                        if pad > 0:
                            seg = np.concatenate([seg, np.full(pad, np.nan, dtype=np.float32)])
                        else:
                            seg = seg[:tone_len_frames]
                    segs.append(seg.astype(np.float32))
                heat_rows.append(np.concatenate(segs, axis=0))
            else:
                heat_rows.append(Z[:, j].astype(np.float32))

    meta = pd.DataFrame(meta_rows)
    H_all = np.vstack(heat_rows).astype(np.float32)

    # Filter groups included in heatmap
    keep = meta["group"].isin(include_groups).values
    meta_k = meta.loc[keep].reset_index(drop=True)
    H_k = H_all[keep, :]

    # Sort by group order, then sort_score desc, then stable tie-breaks
    group_order = cfg.get("group_order", ["EarlyOnly", "Overlap", "LateOnly"])
    group_order = [str(g) for g in group_order if str(g) in include_groups]

    parts = []
    idx_parts = []
    meta_k["_row"] = np.arange(len(meta_k))

    for g in group_order:
        sub = meta_k[meta_k["group"] == g].copy()
        sub = sub.sort_values(["sort_score", "animal_id", "cell_id"], ascending=[False, True, True])
        parts.append(sub)
        idx_parts.append(sub["_row"].values.astype(int))

    if parts:
        meta_sorted = pd.concat(parts, ignore_index=True)
        row_idx = np.concatenate(idx_parts) if idx_parts else np.arange(len(meta_k))
    else:
        meta_sorted = meta_k.copy()
        row_idx = np.arange(len(meta_k))

    H_sorted = H_k[row_idx, :]

    # Boundaries at end of each tone segment (for separators)
    if tone_only:
        boundaries = np.array([(i + 1) * tone_len_frames for i in range(n_tones)], dtype=int)
    else:
        boundaries = np.array([], dtype=int)

    if dry_run:
        print(f"[DRY RUN] Would write heatmap: {H_sorted.shape}, n_tones={n_tones}, tone_len_frames={tone_len_frames}")
        return dict(meta=meta_sorted, heatmap=H_sorted, boundaries=boundaries)

    # Save NPZ + CSV
    np.savez_compressed(npz_out, heatmap=H_sorted.astype(np.float32), boundaries=boundaries.astype(int))
    meta_sorted.drop(columns=["_row"], errors="ignore").to_csv(csv_out, index=False)

    # Plot with group-specific gradients + clipping
    vmin, vmax = clip_min, clip_max
    Hc = np.clip(H_sorted, vmin, vmax)
    norm = (Hc - vmin) / (vmax - vmin)

    N, T = H_sorted.shape
    rgb = np.empty((N, T, 3), dtype=np.float32)
    groups = meta_sorted["group"].astype(str).tolist()
    for i, g in enumerate(groups):
        b = base.get(g, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        a = norm[i, :].reshape(-1, 1)
        rgb[i, :, :] = white * (1 - a) + b * a

    fig_h = min(fig_h_max, max(fig_h_min, N / cells_per_inch))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgb, aspect="auto", interpolation="nearest")

    if boundaries.size >= 2:
        for b in boundaries[1:-1]:
            ax.axvline(b - 0.5, linewidth=vline_lw, color="k", alpha=vline_alpha)

    group_sizes = [int((meta_sorted["group"] == g).sum()) for g in group_order]
    y = 0
    for sz in group_sizes[:-1]:
        y += sz
        ax.axhline(y - 0.5, linewidth=hline_lw, color="k", alpha=hline_alpha)

    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(png_out, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Bundle outputs
    with zipfile.ZipFile(bundle_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_out, arcname=os.path.basename(npz_out))
        zf.write(csv_out, arcname=os.path.basename(csv_out))
        zf.write(png_out, arcname=os.path.basename(png_out))

    with zipfile.ZipFile(bundle_png_out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(png_out, arcname=os.path.basename(png_out))

    # metadata
    run_meta = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "out_prefix": out_prefix,
        "out_dir": out_dir,
        "work_dir": work_dir,
        "input_zips": [str(z) for z in input_zips],
        "input_zip_sha256": {str(z): sha256_file(str(z)) for z in input_zips if os.path.exists(str(z))},
        "excluded_files": excluded,
        "derived": {
            "n_animals": len(records),
            "n_tones_used": int(n_tones),
            "tone_len_frames": int(tone_len_frames),
            "tone_len_policy": tone_len_policy,
            "tone_only": bool(tone_only),
            "heatmap_shape": [int(x) for x in H_sorted.shape],
        },
        "params": cfg,
        "outputs": {
            "npz": npz_out,
            "csv": csv_out,
            "png": png_out,
            "bundle": bundle_out,
            "bundle_png_only": bundle_png_out,
        }
    }
    with open(meta_out, "w") as f:
        json.dump(run_meta, f, indent=2)

    print("Wrote:")
    print(" ", npz_out)
    print(" ", csv_out)
    print(" ", png_out)
    print(" ", bundle_out)
    print(" ", meta_out)

    return run_meta

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config.")
    ap.add_argument("--out_prefix", default=None, help="Override out_prefix in config.")
    ap.add_argument("--out_dir", default=None, help="Override out_dir in config.")
    ap.add_argument("--work_dir", default=None, help="Override work_dir in config.")
    ap.add_argument("--dry_run", action="store_true", help="Parse + compute sizes but do not write files.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    if args.out_prefix is not None: overrides["out_prefix"] = args.out_prefix
    if args.out_dir is not None: overrides["out_dir"] = args.out_dir
    if args.work_dir is not None: overrides["work_dir"] = args.work_dir
    cfg = deep_update(cfg, overrides)

    build_heatmap(cfg, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
