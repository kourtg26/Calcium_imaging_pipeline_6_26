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
    m = re.match(r"^(animal\d+)", base, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|_)(\d{1,5})(?:_|$)", base)
    if m:
        return m.group(1)
    return base.split("_")[0]

def find_time_col(df: pd.DataFrame):
    cols = list(df.columns)
    exact = ["Time_s", "Time (s)", "Time(s)", "Time"]
    for c in exact:
        if c in cols:
            return c
    for c in cols:
        if "time" in str(c).lower():
            return c
    if len(cols) > 0:
        c0 = cols[0]
        lc0 = str(c0).strip().lower()
        if lc0 in ("", "unnamed: 0", "unnamed:0") or "unnamed" in lc0:
            v = pd.to_numeric(df[c0], errors="coerce").dropna().values
            if v.size >= 3:
                dv = np.diff(v.astype(np.float64))
                good = np.isfinite(dv) & (dv > 0)
                if good.mean() >= 0.8:
                    return c0
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

def resample_segment(seg: np.ndarray, out_len: int) -> np.ndarray:
    """Linearly resample 1D segment to out_len while handling NaNs."""
    if out_len <= 0:
        return np.empty((0,), dtype=np.float32)
    x = np.asarray(seg, dtype=np.float32).reshape(-1)
    if x.size == out_len:
        return x
    if x.size == 0:
        return np.full(out_len, np.nan, dtype=np.float32)
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return np.full(out_len, np.nan, dtype=np.float32)
    if valid.sum() == 1:
        return np.full(out_len, float(x[valid][0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=x.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False, dtype=np.float64)
    y = np.interp(x_new, x_old[valid], x[valid].astype(np.float64))
    return y.astype(np.float32)

def normalize_animal_key(v) -> str:
    s = str(v).strip().lower()
    m = re.fullmatch(r"animal0*(\d+)", s)
    if m:
        return f"animal{int(m.group(1))}"
    if re.fullmatch(r"\d+", s):
        return str(int(s))
    return s

def animal_digit_key(v):
    s = str(v).strip().lower()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return str(int(m.group(1)))

def normalize_cell_key(v) -> str:
    s = str(v).strip().upper()
    m = re.fullmatch(r"UNDECIDED(?:\.(\d+))?", s)
    if m:
        idx = 0 if m.group(1) is None else int(m.group(1))
        return f"IDX.{idx}"
    m = re.fullmatch(r"C0*(\d+)", s)
    if m:
        return f"IDX.{int(m.group(1))}"
    m = re.fullmatch(r"0*(\d+)", s)
    if m:
        return f"IDX.{int(m.group(1))}"
    return s

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
    min_hits_per_period = max(1, int(oe.get("min_hits_per_period", 1)))
    dz_reference = str(oe.get("dz_reference", "onset_sample"))  # onset_sample | pre_window
    pre_win_s = float(oe.get("pre_win_s", 1.0))  # only used if pre_window
    classification_mode = str(oe.get("classification_mode", "legacy_delta_run"))

    # grouping params
    grp = cfg.get("groups", {}) or {}
    early_n = int(grp.get("early_n", 3))
    late_n  = int(grp.get("late_n", 3))

    # heatmap params
    hm = cfg.get("heatmap", {}) or {}
    tone_only = bool(hm.get("tone_only", True))
    tone_len_policy = str(hm.get("tone_len_policy", "min_across_all"))  # min_across_all | fixed_frames | fixed_seconds | per_tone_median
    fixed_tone_len_frames = hm.get("fixed_tone_len_frames", None)
    fixed_tone_len_seconds = hm.get("fixed_tone_len_seconds", None)
    fixed_seconds_exact = bool(hm.get("fixed_seconds_exact", False))
    global_dt_seconds = hm.get("global_dt_seconds", None)
    delta_from_tone_onset = bool(hm.get("delta_from_tone_onset", False))
    increase_only = bool(hm.get("increase_only", False))
    sort_within_group = str(hm.get("sort_within_group", "sort_score_desc"))  # sort_score_desc | mean_tone_z_desc

    # cell id handling
    cid_cfg = cfg.get("cell_id", {}) or {}
    pad_numeric_ids_to = cid_cfg.get("pad_numeric_ids_to", None)
    if pad_numeric_ids_to is not None:
        pad_numeric_ids_to = int(pad_numeric_ids_to)

    # optional: force inclusion/group labels from historical metadata
    wl_cfg = cfg.get("cell_whitelist", {}) or {}
    whitelist_csv = wl_cfg.get("csv_path", None)
    whitelist_animal_col = str(wl_cfg.get("animal_col", "animal"))
    whitelist_cell_col = str(wl_cfg.get("cell_col", "cell"))
    whitelist_group_col = str(wl_cfg.get("group_col", "group"))
    whitelist_source_col = str(wl_cfg.get("source_file_col", "source_file"))
    whitelist_override_group = bool(wl_cfg.get("override_group", True))
    whitelist_use_digit_fallback = bool(wl_cfg.get("use_animal_digit_fallback", True))
    whitelist_include_extra_animals = wl_cfg.get("include_extra_animals", []) or []
    whitelist_include_extra_animals = [str(x) for x in whitelist_include_extra_animals]
    whitelist_include_extra_keys = {normalize_animal_key(x) for x in whitelist_include_extra_animals}

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
    targets_name = f"{out_prefix}_toneOnly_resample_targets.csv"

    npz_out = os.path.join(out_dir, npz_name)
    csv_out = os.path.join(out_dir, csv_name)
    png_out = os.path.join(out_dir, png_name)
    bundle_out = os.path.join(out_dir, bundle_name)
    bundle_png_out = os.path.join(out_dir, bundle_png_only_name)
    meta_out = os.path.join(out_dir, meta_name)
    targets_out = os.path.join(out_dir, targets_name)

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
    csv_paths = sorted(glob.glob(os.path.join(work_dir, "**", "*.csv"), recursive=True))
    if not csv_paths:
        raise RuntimeError(f"No CSVs found in work_dir: {work_dir}")

    # Load per-animal records
    records = []
    excluded = []
    for fp in csv_paths:
        try:
            df = pd.read_csv(fp)
            df.columns = [str(c).strip() for c in df.columns]
            time_col = find_time_col(df)
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
            raw_cell_labels = []
            for j, c in enumerate(trace_cols):
                name = str(c).strip()
                raw_cell_labels.append(name)
                m_c = re.fullmatch(r"C(\d+)", name, flags=re.IGNORECASE)
                if m_c:
                    digits = m_c.group(1)
                    if pad_numeric_ids_to is not None and pad_numeric_ids_to > 0:
                        digits = digits.zfill(pad_numeric_ids_to)
                    cell_ids.append(f"C{digits}")
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
            if len(bouts) < max(early_n, late_n):
                excluded.append((fp, f"too_few_tones_{len(bouts)}"))
                continue

            animal_id = infer_animal_id(fp)
            if not re.search(r"\d", str(animal_id)):
                excluded.append((fp, "no_numeric_animal_id"))
                continue
            records.append(dict(
                animal_id=animal_id,
                file=fp,
                t=t,
                dt=dt,
                tone=tone,
                freeze=freeze,
                X=X,
                cell_ids=cell_ids,
                raw_cell_labels=raw_cell_labels,
                bouts=bouts,
            ))
        except Exception as e:
            excluded.append((fp, f"read_error:{type(e).__name__}:{e}"))

    if not records:
        raise RuntimeError("No valid CSVs found (need time + binary tone flag + trace columns).")

    # Harmonize tone count across animals
    n_tones = min(len(r["bouts"]) for r in records)

    # Determine tone length in frames
    tone_target_frames = None
    tone_target_seconds = None
    dt_for_resample = None
    if tone_len_policy == "min_across_all":
        tone_len_frames = min((e - s) for r in records for (s, e) in r["bouts"][:n_tones])
    elif tone_len_policy == "fixed_frames":
        if fixed_tone_len_frames is None:
            raise ValueError("fixed_tone_len_frames required when tone_len_policy=fixed_frames.")
        tone_len_frames = int(fixed_tone_len_frames)
    elif tone_len_policy == "fixed_seconds":
        if fixed_tone_len_seconds is None:
            raise ValueError("fixed_tone_len_seconds required when tone_len_policy=fixed_seconds.")
        # Target timeline uses minimum dt across animals.
        dt_min = min(r["dt"] for r in records)
        dt_for_resample = float(dt_min)
        tone_len_frames = frames_from_seconds(dt_min, float(fixed_tone_len_seconds))
    elif tone_len_policy == "per_tone_median":
        dt_for_resample = float(np.median([r["dt"] for r in records])) if global_dt_seconds is None else float(global_dt_seconds)
        tone_target_seconds = []
        tone_target_frames = []
        for k in range(n_tones):
            durations_s = []
            for r in records:
                s, e = r["bouts"][k]
                durations_s.append((e - s) * r["dt"])
            med_s = float(np.median(durations_s))
            tone_target_seconds.append(med_s)
            tone_target_frames.append(frames_from_seconds(dt_for_resample, med_s))
        tone_len_frames = None
    else:
        raise ValueError(f"Unknown tone_len_policy: {tone_len_policy}")

    if tone_len_policy != "per_tone_median" and tone_len_frames <= 0:
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
        raw_cell_labels = r.get("raw_cell_labels", cell_ids)

        Z = zscore_cols(X)  # (T x Ncells)

        post_frames = frames_from_seconds(dt, post_win_s)
        pre_frames  = frames_from_seconds(dt, pre_win_s)

        early_idx = list(range(min(early_n, n_tones)))
        late_idx  = list(range(max(0, n_tones - late_n), n_tones))

        for j, cid in enumerate(cell_ids):
            raw_label = raw_cell_labels[j] if j < len(raw_cell_labels) else cid
            early_peak = -np.inf
            late_peak  = -np.inf

            early_hits = []
            late_hits  = []

            for k in early_idx:
                s, _e = bouts[k]
                idx0 = s
                start = idx0 + 1  # use (0, post_win_s] after onset
                w_end = min(Z.shape[0], idx0 + post_frames + 1)

                if dz_reference == "onset_sample":
                    ref = Z[idx0, j]
                elif dz_reference == "pre_window":
                    w0 = max(0, idx0 - pre_frames)
                    ref = float(np.nanmean(Z[w0:idx0+1, j]))
                else:
                    raise ValueError(f"Unknown dz_reference: {dz_reference}")

                z_post = Z[start:w_end, j]
                dz = z_post - ref
                if dz.size:
                    early_peak = max(early_peak, float(np.nanmax(dz)))
                    if classification_mode == "legacy_delta_run":
                        early_hits.append(has_run_ge(dz, dz_thresh, consec_frames))
                    elif classification_mode == "delta_pos_and_abs_z_run":
                        early_hits.append(bool(np.any(dz > 0.0) and has_run_ge(z_post, dz_thresh, consec_frames)))
                    else:
                        raise ValueError(f"Unknown classification_mode: {classification_mode}")
                else:
                    early_hits.append(False)

            for k in late_idx:
                s, _e = bouts[k]
                idx0 = s
                start = idx0 + 1  # use (0, post_win_s] after onset
                w_end = min(Z.shape[0], idx0 + post_frames + 1)

                if dz_reference == "onset_sample":
                    ref = Z[idx0, j]
                elif dz_reference == "pre_window":
                    w0 = max(0, idx0 - pre_frames)
                    ref = float(np.nanmean(Z[w0:idx0+1, j]))
                else:
                    raise ValueError(f"Unknown dz_reference: {dz_reference}")

                z_post = Z[start:w_end, j]
                dz = z_post - ref
                if dz.size:
                    late_peak = max(late_peak, float(np.nanmax(dz)))
                    if classification_mode == "legacy_delta_run":
                        late_hits.append(has_run_ge(dz, dz_thresh, consec_frames))
                    elif classification_mode == "delta_pos_and_abs_z_run":
                        late_hits.append(bool(np.any(dz > 0.0) and has_run_ge(z_post, dz_thresh, consec_frames)))
                    else:
                        raise ValueError(f"Unknown classification_mode: {classification_mode}")
                else:
                    late_hits.append(False)

            early_resp = bool(sum(1 for h in early_hits if h) >= min_hits_per_period)
            late_resp  = bool(sum(1 for h in late_hits if h) >= min_hits_per_period)

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

            mean_tone_z = np.nan
            if tone_only:
                segs = []
                tone_means = []
                for k in range(n_tones):
                    s, e = bouts[k]
                    if tone_len_policy == "per_tone_median":
                        seg = Z[s:e, j]
                        if delta_from_tone_onset:
                            ref = Z[s, j] if (0 <= s < Z.shape[0]) else np.nan
                            seg = seg - ref
                            if increase_only:
                                seg = np.maximum(seg, 0.0)
                        tone_means.append(float(np.nanmean(seg)) if seg.size else np.nan)
                        seg = resample_segment(seg, int(tone_target_frames[k]))
                    elif tone_len_policy == "fixed_seconds" and fixed_seconds_exact:
                        src_len = frames_from_seconds(dt, float(fixed_tone_len_seconds))
                        seg = Z[s:s + src_len, j]
                        if delta_from_tone_onset:
                            ref = Z[s, j] if (0 <= s < Z.shape[0]) else np.nan
                            seg = seg - ref
                            if increase_only:
                                seg = np.maximum(seg, 0.0)
                        tone_means.append(float(np.nanmean(seg)) if seg.size else np.nan)
                        seg = resample_segment(seg, tone_len_frames)
                    else:
                        seg = Z[s:s + tone_len_frames, j]
                        if delta_from_tone_onset:
                            ref = Z[s, j] if (0 <= s < Z.shape[0]) else np.nan
                            seg = seg - ref
                            if increase_only:
                                seg = np.maximum(seg, 0.0)
                        tone_means.append(float(np.nanmean(seg)) if seg.size else np.nan)
                        # pad/truncate to tone_len_frames
                        if seg.shape[0] != tone_len_frames:
                            pad = tone_len_frames - seg.shape[0]
                            if pad > 0:
                                seg = np.concatenate([seg, np.full(pad, np.nan, dtype=np.float32)])
                            else:
                                seg = seg[:tone_len_frames]
                    segs.append(seg.astype(np.float32))
                heat_rows.append(np.concatenate(segs, axis=0))
                mean_tone_z = float(np.nanmean(tone_means)) if len(tone_means) > 0 else np.nan
            else:
                heat_rows.append(Z[:, j].astype(np.float32))

            meta_rows.append(dict(
                animal_id=animal,
                cell_id=cid,
                raw_cell_label=raw_label,
                group=group,
                early_peak_dZ=early_peak if np.isfinite(early_peak) else np.nan,
                late_peak_dZ=late_peak if np.isfinite(late_peak) else np.nan,
                sort_score=sort_score if np.isfinite(sort_score) else np.nan,
                mean_tone_z=mean_tone_z,
                dt=dt,
                source_file=os.path.basename(r["file"]),
            ))

    meta = pd.DataFrame(meta_rows)
    H_all = np.vstack(heat_rows).astype(np.float32)

    whitelist_stats = {
        "enabled": bool(whitelist_csv),
        "csv_path": str(whitelist_csv) if whitelist_csv else None,
        "include_extra_animals": whitelist_include_extra_animals,
        "whitelist_rows": None,
        "matched_rows": None,
        "unmatched_rows": None,
        "kept_extra_rows": None,
        "matched_by_exact": None,
        "matched_by_digit_fallback": None,
    }
    if whitelist_csv:
        wl_path = os.path.expanduser(str(whitelist_csv))
        if not os.path.exists(wl_path):
            raise FileNotFoundError(f"cell_whitelist.csv_path not found: {wl_path}")
        wl_df = pd.read_csv(wl_path)
        for c in (whitelist_animal_col, whitelist_cell_col):
            if c not in wl_df.columns:
                raise ValueError(f"Whitelist missing required column: {c}")
        if whitelist_override_group and whitelist_group_col not in wl_df.columns:
            raise ValueError(f"Whitelist missing required group column: {whitelist_group_col}")

        wl_use = wl_df[[whitelist_animal_col, whitelist_cell_col] + ([whitelist_group_col] if whitelist_group_col in wl_df.columns else [])].copy()
        wl_use["_animal_key"] = wl_use[whitelist_animal_col].map(normalize_animal_key)
        wl_use["_animal_digit"] = wl_use[whitelist_animal_col].map(animal_digit_key)
        wl_use["_cell_key"] = wl_use[whitelist_cell_col].map(normalize_cell_key)
        if whitelist_source_col in wl_df.columns:
            wl_use["_source_key"] = wl_df[whitelist_source_col].map(lambda x: os.path.basename(str(x)).strip().lower())
        else:
            wl_use["_source_key"] = np.nan

        exact_source_map = {}
        digit_source_map = {}
        exact_map = {}
        digit_map = {}
        for _, wr in wl_use.iterrows():
            g = wr[whitelist_group_col] if whitelist_group_col in wl_use.columns else "__KEEP__"
            if pd.notna(wr["_source_key"]):
                exact_source_map[(wr["_animal_key"], wr["_source_key"], wr["_cell_key"])] = g
                if pd.notna(wr["_animal_digit"]):
                    digit_source_map[(wr["_animal_digit"], wr["_source_key"], wr["_cell_key"])] = g
            exact_map[(wr["_animal_key"], wr["_cell_key"])] = g
            if pd.notna(wr["_animal_digit"]):
                digit_map[(wr["_animal_digit"], wr["_cell_key"])] = g

        meta["_animal_key"] = meta["animal_id"].map(normalize_animal_key)
        meta["_animal_digit"] = meta["animal_id"].map(animal_digit_key)
        cell_match_col = "raw_cell_label" if "raw_cell_label" in meta.columns else "cell_id"
        meta["_cell_key"] = meta[cell_match_col].map(normalize_cell_key)
        meta["_source_key"] = meta["source_file"].map(lambda x: os.path.basename(str(x)).strip().lower())

        matched_groups = []
        matched_exact = 0
        matched_source_exact = 0
        matched_source_fallback = 0
        matched_fallback = 0
        for _, mr in meta.iterrows():
            g = None
            k_source_exact = (mr["_animal_key"], mr["_source_key"], mr["_cell_key"])
            k_exact = (mr["_animal_key"], mr["_cell_key"])
            if k_source_exact in exact_source_map:
                g = exact_source_map[k_source_exact]
                matched_source_exact += 1
            elif k_exact in exact_map:
                g = exact_map[k_exact]
                matched_exact += 1
            elif whitelist_use_digit_fallback and pd.notna(mr["_animal_digit"]):
                k_source_fallback = (mr["_animal_digit"], mr["_source_key"], mr["_cell_key"])
                k_fallback = (mr["_animal_digit"], mr["_cell_key"])
                if k_source_fallback in digit_source_map:
                    g = digit_source_map[k_source_fallback]
                    matched_source_fallback += 1
                elif k_fallback in digit_map:
                    g = digit_map[k_fallback]
                    matched_fallback += 1
            matched_groups.append(g)

        meta["_whitelist_group"] = matched_groups
        matched_mask = meta["_whitelist_group"].notna().values
        keep_extra = meta["_animal_key"].isin(whitelist_include_extra_keys).values if whitelist_include_extra_keys else np.zeros(len(meta), dtype=bool)
        extra_only_mask = keep_extra & (~matched_mask)
        keep_wl = matched_mask | keep_extra
        meta = meta.loc[keep_wl].reset_index(drop=True)
        H_all = H_all[keep_wl, :]
        if whitelist_override_group:
            override_mask = meta["_whitelist_group"].notna().values
            meta.loc[override_mask, "group"] = meta.loc[override_mask, "_whitelist_group"].astype(str).values

        whitelist_stats.update({
            "whitelist_rows": int(len(wl_use)),
            "matched_rows": int(np.sum(matched_mask)),
            "unmatched_rows": int(np.sum(~matched_mask)),
            "kept_extra_rows": int(np.sum(extra_only_mask)),
            "matched_by_source_exact": int(matched_source_exact),
            "matched_by_exact": int(matched_exact),
            "matched_by_source_digit_fallback": int(matched_source_fallback),
            "matched_by_digit_fallback": int(matched_fallback),
        })

        meta = meta.drop(columns=["_animal_key", "_animal_digit", "_cell_key", "_source_key", "_whitelist_group"], errors="ignore")

    # Filter groups included in heatmap
    keep = meta["group"].isin(include_groups).values
    meta_k = meta.loc[keep].reset_index(drop=True)
    H_k = H_all[keep, :]

    # Sort by group order, then chosen within-group metric desc, then stable tie-breaks
    group_order = cfg.get("group_order", ["EarlyOnly", "Overlap", "LateOnly"])
    group_order = [str(g) for g in group_order if str(g) in include_groups]

    parts = []
    idx_parts = []
    meta_k["_row"] = np.arange(len(meta_k))

    for g in group_order:
        sub = meta_k[meta_k["group"] == g].copy()
        if sort_within_group == "mean_tone_z_desc":
            sub = sub.sort_values(["mean_tone_z", "animal_id", "cell_id"], ascending=[False, True, True])
        else:
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
        if tone_len_policy == "per_tone_median":
            boundaries = np.cumsum(np.asarray(tone_target_frames, dtype=int))
        else:
            boundaries = np.array([(i + 1) * tone_len_frames for i in range(n_tones)], dtype=int)
    else:
        boundaries = np.array([], dtype=int)

    if dry_run:
        print(f"[DRY RUN] Would write heatmap: {H_sorted.shape}, n_tones={n_tones}, tone_len_policy={tone_len_policy}")
        return dict(meta=meta_sorted, heatmap=H_sorted, boundaries=boundaries)

    # Save NPZ + CSV
    np.savez_compressed(npz_out, heatmap=H_sorted.astype(np.float32), boundaries=boundaries.astype(int))
    meta_sorted.drop(columns=["_row"], errors="ignore").to_csv(csv_out, index=False)
    if tone_target_frames is not None and tone_target_seconds is not None:
        pd.DataFrame({
            "tone": np.arange(1, n_tones + 1, dtype=int),
            "target_duration_s": np.asarray(tone_target_seconds, dtype=float),
            "target_n_samples": np.asarray(tone_target_frames, dtype=int),
            "global_dt_s": float(dt_for_resample) if dt_for_resample is not None else np.nan,
        }).to_csv(targets_out, index=False)

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
        if os.path.exists(targets_out):
            zf.write(targets_out, arcname=os.path.basename(targets_out))
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
            "tone_len_frames": int(tone_len_frames) if tone_len_frames is not None else None,
            "tone_len_policy": tone_len_policy,
            "tone_target_frames": [int(x) for x in tone_target_frames] if tone_target_frames is not None else None,
            "tone_target_seconds": [float(x) for x in tone_target_seconds] if tone_target_seconds is not None else None,
            "global_dt_seconds": float(dt_for_resample) if dt_for_resample is not None else None,
            "fixed_seconds_exact": bool(fixed_seconds_exact),
            "delta_from_tone_onset": bool(delta_from_tone_onset),
            "increase_only": bool(increase_only),
            "classification_mode": classification_mode,
            "sort_within_group": sort_within_group,
            "tone_only": bool(tone_only),
            "heatmap_shape": [int(x) for x in H_sorted.shape],
            "cell_whitelist": whitelist_stats,
        },
        "params": cfg,
        "outputs": {
            "npz": npz_out,
            "csv": csv_out,
            "resample_targets_csv": targets_out if os.path.exists(targets_out) else None,
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
