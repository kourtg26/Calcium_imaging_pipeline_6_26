#!/usr/bin/env python3
"""
ext2_ret_transition_pipeline.py

Purpose
-------
1) Build Ext2 onset-evoked classification from RAW trace CSVs using the same rule as Ext2:
   - ΔZ(t) = Z(t) - Z(t0) where t0 is tone onset sample
   - responsive if ΔZ >= dz_thresh for >= consec_frames within 0..post_win_s after onset
   - early window = first early_n tones; late window = last late_n tones
   - early/late responsive requires >= min_hits_per_period hit tones in that window
   - classes: EarlyOnly / LateOnly / Overlap / Neither

2) Merge Ext2 and Retrieval classifications by (animal_id, cell_id) assuming registration preserves cell_id strings,
   compute transition matrices and export Sankey/alluvial plots (SVG/PDF/PNG) + CSV tables.

Inputs
------
- Retrieval classification zip (already produced): ret_classes_proportions_only.zip
- Ext2 raw trace zips: one or more .zip containing per-animal CSVs

Expected raw CSV columns
------------------------
- Time: one of {Time_s, Time (s), Time(s), Time} or any column containing "time"
- Tone flag: a binary (0/1) column whose name contains "tone" or "cs", e.g. ToneFlag/WithinTone
- Optional Freeze flag: binary 0/1 column containing "freez"
- Cell traces: columns named like C001, C002, ... (preferred). If not, numeric columns w/ ~complete data are used.

Outputs
-------
- ext2_classes_proportions_only.zip (per-animal class CSVs + ext2_onset_evoked_proportions_by_animal.csv)
- Ext2toRet_transition_counts_cellWeighted.csv
- Ext2toRet_transition_props_animalWeighted.csv
- Sankey/alluvial plots (overall + by sex, cell-weighted and animal-weighted)

Usage
-----
python ext2_ret_transition_pipeline.py \
  --ret_zip /mnt/data/ret_classes_proportions_only.zip \
  --ext2_raw_zips /mnt/data/ext2_raw_trace_files_part1.zip /mnt/data/ext2_raw_trace_files_part2.zip \
  --out_dir /mnt/data \
  --work_dir /mnt/data/_ext2_ret_transition_work
"""

import os, re, glob, zipfile, json, argparse, hashlib, datetime, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from scipy import stats
try:
    import plotly.graph_objects as go
except Exception:
    go = None

# --------------------------
# User sex mapping (edit if needed)
# --------------------------
FEMALE = {"4","6","8","10","10488","10489","10490","10491","11794"}
MALE   = {"1","2","3","5","9","10492","10481","10482","11799"}

VALID = ["EarlyOnly","Overlap","LateOnly","Neither"]

COLOR = {
    "EarlyOnly": (1.0, 0.0, 0.0),
    "Overlap":   (0.5, 0.5, 0.5),
    "LateOnly":  (0.0, 0.0, 1.0),
    "Neither":   (0.75,0.75,0.75),
}

def norm_animal_id(a: str) -> str:
    a = str(a)
    if a.lower().startswith("animal"):
        a = a.lower().replace("animal","", 1)
    if a.lower() == "cell":
        return "10492"
    return a

def sex_of(animal_id: str) -> str:
    n = norm_animal_id(animal_id)
    if n in FEMALE: return "Female"
    if n in MALE:   return "Male"
    return "Unknown"

def normalize_cell_id_any(cid: str) -> str:
    """
    Normalize cell IDs to numeric strings:
    - "C1", "C01", "C001" -> "1"
    - "1", "01", "001" -> "1"
    - "undecided.3" -> "3"
    """
    s = str(cid).strip()
    if s == "":
        return None
    ms = re.findall(r"(\d+)", s)
    if ms:
        return str(int(ms[-1]))
    return None

# --------------------------
# IO helpers
# --------------------------
def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def ensure_clean_dir(p: str) -> str:
    if os.path.isdir(p):
        shutil.rmtree(p)
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def extract_all(zips, outdir):
    for zp in zips:
        if not os.path.exists(zp):
            raise FileNotFoundError(f"Missing zip: {zp}")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(outdir)

def infer_animal_id(filename: str) -> str:
    base = os.path.basename(filename)
    # common animal naming (animalX_*)
    m = re.search(r"animal[_-]?(\d+)", base, re.IGNORECASE)
    if m:
        return norm_animal_id(m.group(1))
    # explicit cell dataset
    if base.lower().startswith("cell"):
        return "10492"
    # filenames that start with numeric animal id (e.g., 10488_ext2_...)
    m = re.match(r"^(\d{4,5})_", base)
    if m:
        return norm_animal_id(m.group(1))
    # fallback: any 4-5 digit id inside filename
    m = re.search(r"(\d{4,5})", base)
    if m:
        return norm_animal_id(m.group(1))
    # last resort: leading token
    m = re.match(r"^([A-Za-z]*\d+)", base)
    if m:
        return norm_animal_id(m.group(1))
    return norm_animal_id(base.split("_")[0])

# --------------------------
# Column detection
# --------------------------
def find_time_col(df: pd.DataFrame):
    cols = list(df.columns)
    exact = ["Time_s", "Time (s)", "Time(s)", "Time"]
    for c in exact:
        if c in cols:
            return c
    for c in cols:
        if "time" in str(c).lower():
            return c
    # fallback: common index-like columns if numeric and monotonic
    for c in cols:
        cname = str(c).strip().lower()
        if cname.startswith("unnamed") or cname == "":
            v = pd.to_numeric(df[c], errors="coerce")
            vv = v.dropna().values
            if vv.size > 2 and np.all(np.diff(vv) >= 0):
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
        if ("tone" in lc or "intone" in lc or lc == "cs") and _is_binary_series(df[c]):
            return c
    return None

def find_freeze_col(df: pd.DataFrame):
    cols = list(df.columns)
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

    trace_cols = []
    for c in cols:
        if is_aux(c): 
            continue
        name = str(c).strip()
        if re.fullmatch(r"C\d+", name):
            trace_cols.append(c)
    if trace_cols:
        return trace_cols

    numeric_cols = []
    for c in cols:
        if is_aux(c): 
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.95 and np.nanstd(s.values) > 1e-12:
            numeric_cols.append(c)
    return numeric_cols

# --------------------------
# Tone bouts / z-scoring / classification
# --------------------------
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

def frames_from_seconds(dt: float, seconds: float) -> int:
    return max(1, int(np.floor(seconds / dt))) if dt > 0 else 1

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

def classify_one_animal(df: pd.DataFrame, dz_thresh: float, consec_frames: int, post_win_s: float, early_n: int, late_n: int, min_hits_per_period: int):
    time_col = find_time_col(df)
    if time_col is None:
        raise ValueError("No time column.")
    tone_col = find_tone_col(df)
    if tone_col is None:
        raise ValueError("No binary tone flag column.")
    freeze_col = find_freeze_col(df)

    trace_cols = select_trace_cols(df, time_col, tone_col, freeze_col)
    if len(trace_cols) == 0:
        raise ValueError("No trace columns found.")

    t = pd.to_numeric(df[time_col], errors="coerce").values.astype(np.float64)
    tt = t[np.isfinite(t)]
    if len(tt) < 3:
        raise ValueError("Bad time vector.")
    dt = float(np.median(np.diff(tt)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Bad dt.")

    tone = pd.to_numeric(df[tone_col], errors="coerce").fillna(0).values.astype(int)
    bouts = tone_bouts_from_flag(tone)
    if len(bouts) < max(early_n, late_n):
        raise ValueError(f"Too few tones: {len(bouts)}")

    X = df[trace_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    Z = zscore_cols(X)  # T x N
    cell_ids = [normalize_cell_id_any(c) for c in trace_cols]

    post_frames = frames_from_seconds(dt, post_win_s)

    n_tones = len(bouts)
    early_idx = list(range(min(early_n, n_tones)))
    late_idx  = list(range(max(0, n_tones - late_n), n_tones))

    out = []
    for j, cid in enumerate(cell_ids):
        early_peak = -np.inf
        late_peak  = -np.inf
        early_hits = []
        late_hits  = []

        for k in early_idx:
            s, _e = bouts[k]
            idx0 = s
            w_end = min(Z.shape[0], idx0 + post_frames + 1)
            ref = Z[idx0, j]
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
            ref = Z[idx0, j]
            dz = Z[idx0:w_end, j] - ref
            if dz.size:
                late_peak = max(late_peak, float(np.nanmax(dz)))
                late_hits.append(has_run_ge(dz, dz_thresh, consec_frames))
            else:
                late_hits.append(False)

        early_resp = int(np.sum(early_hits)) >= int(min_hits_per_period)
        late_resp  = int(np.sum(late_hits)) >= int(min_hits_per_period)

        if early_resp and (not late_resp):
            cls = "EarlyOnly"
        elif late_resp and (not early_resp):
            cls = "LateOnly"
        elif early_resp and late_resp:
            cls = "Overlap"
        else:
            cls = "Neither"

        out.append({
            "cell_id": cid,
            "class": cls,
            "early_peak_dZ": (early_peak if np.isfinite(early_peak) else np.nan),
            "late_peak_dZ": (late_peak if np.isfinite(late_peak) else np.nan),
            "dt": dt,
            "n_tones": n_tones
        })

    out_df = pd.DataFrame(out)
    return out_df

# --------------------------
# Sankey/alluvial plotting (matplotlib, vector-friendly)
# --------------------------
def _stack_positions(order, totals):
    y0={}
    cur=0.0
    for k in order:
        h=float(totals.get(k,0.0))
        y0[k]=cur
        cur += h
    return y0, cur

def alluvial_plot(
    link_df: pd.DataFrame,
    value_col: str,
    title: str,
    out_base: str,
    left_order=VALID,
    right_order=VALID,
    color_by="src",
    left_label="Ext2",
    right_label="Ret",
    show_pct_labels=False,
):
    """
    Simple alluvial: left nodes to right nodes, normalized to total=1 for vertical scale.
    Saves: out_base + .pdf/.svg/.png
    """
    df = link_df.copy()
    df = df[df[value_col] > 0].copy()
    if df.empty:
        raise ValueError("No positive links to plot.")

    total = float(df[value_col].sum())
    df["_h"] = df[value_col] / total

    left_tot = df.groupby("src")["_h"].sum().reindex(left_order).fillna(0.0).to_dict()
    right_tot = df.groupby("tgt")["_h"].sum().reindex(right_order).fillna(0.0).to_dict()

    left_y0, H = _stack_positions(left_order, left_tot)
    right_y0, _ = _stack_positions(right_order, right_tot)

    loff = {k: 0.0 for k in left_order}
    roff = {k: 0.0 for k in right_order}

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    xL0, xL1 = 0.12, 0.18
    xR0, xR1 = 0.82, 0.88
    x0, x1 = xL1, xR0
    c1, c2 = 0.45, 0.55

    df = df.sort_values(["src", "tgt"]).reset_index(drop=True)

    for _, r in df.iterrows():
        src, tgt, h = r["src"], r["tgt"], float(r["_h"])
        y0L = left_y0[src] + loff[src]
        y1L = y0L + h
        y0R = right_y0[tgt] + roff[tgt]
        y1R = y0R + h
        loff[src] += h
        roff[tgt] += h

        verts = [
            (x0, y0L), (c1, y0L), (c2, y0R), (x1, y0R),
            (x1, y1R), (c2, y1R), (c1, y1L), (x0, y1L),
            (x0, y0L),
        ]
        codes = [
            MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]

        col_key = src if color_by == "src" else tgt
        col = COLOR.get(col_key, (0.2, 0.2, 0.2))
        patch = PathPatch(
            MplPath(verts, codes),
            facecolor=col,
            alpha=0.35,
            edgecolor="black",
            linewidth=0.25,
        )
        ax.add_patch(patch)
        if show_pct_labels and h >= 0.03:
            ym = 0.25 * (y0L + y1L + y0R + y1R)
            ax.text(0.5, ym, f"{100*h:.1f}%", fontsize=7, ha="center", va="center", color="black")

    for k in left_order:
        h = left_tot.get(k, 0.0)
        ax.add_patch(Rectangle((xL0, left_y0[k]), xL1 - xL0, h, facecolor=COLOR.get(k, (0.5, 0.5, 0.5)), alpha=0.85, edgecolor="none"))
        ax.text(xL0 - 0.02, left_y0[k] + h / 2, f"{left_label} {k}", ha="right", va="center", fontsize=10)

    for k in right_order:
        h = right_tot.get(k, 0.0)
        ax.add_patch(Rectangle((xR0, right_y0[k]), xR1 - xR0, h, facecolor=COLOR.get(k, (0.5, 0.5, 0.5)), alpha=0.85, edgecolor="none"))
        ax.text(xR1 + 0.02, right_y0[k] + h / 2, f"{right_label} {k}", ha="left", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".svg")
    fig.savefig(out_base + ".png", dpi=300)
    plt.close(fig)

def sankey_plot_html(link_df: pd.DataFrame, value_col: str, title: str, out_html: str, left_label="Ext2", right_label="Ret"):
    if go is None:
        return
    df = link_df.copy()
    df = df[df[value_col] > 0].copy()
    if df.empty:
        return
    left_nodes = VALID.copy()
    right_nodes = VALID.copy()
    labels = [f"{left_label} {k}" for k in left_nodes] + [f"{right_label} {k}" for k in right_nodes]
    left_idx = {k: i for i, k in enumerate(left_nodes)}
    right_idx = {k: i + len(left_nodes) for i, k in enumerate(right_nodes)}
    total = float(df[value_col].sum())

    srcs, tgts, vals, cols, custom = [], [], [], [], []
    for _, r in df.iterrows():
        s = str(r["src"])
        t = str(r["tgt"])
        v = float(r[value_col])
        if v <= 0:
            continue
        srcs.append(left_idx[s])
        tgts.append(right_idx[t])
        vals.append(v)
        rgb = COLOR.get(s, (0.2, 0.2, 0.2))
        cols.append(f"rgba({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)},0.45)")
        custom.append(100.0 * (v / total))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=12,
                    thickness=16,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=[f"rgba({int(COLOR.get(k, (0.5,0.5,0.5))[0]*255)},{int(COLOR.get(k, (0.5,0.5,0.5))[1]*255)},{int(COLOR.get(k, (0.5,0.5,0.5))[2]*255)},0.85)" for k in left_nodes + right_nodes],
                ),
                link=dict(
                    source=srcs,
                    target=tgts,
                    value=vals,
                    color=cols,
                    line=dict(color="black", width=0.3),
                    customdata=custom,
                    hovertemplate="%{source.label} → %{target.label}<br>Value=%{value:.3g}<br>Percent=%{customdata:.1f}%<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(title_text=title, font_size=11)
    fig.write_html(out_html, include_plotlyjs="cdn")

# --------------------------
# Transition computation
# --------------------------
def load_class_tables_from_zip(zp: str, session_tag: str, outdir: str):
    """Extract zip to outdir/session_tag and load per-animal class csvs into dict[animal]->df."""
    sess_dir = os.path.join(outdir, session_tag)
    ensure_clean_dir(sess_dir)
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(sess_dir)
    # Special-case Retrieval bundle (single consolidated CSV)
    if session_tag.lower() == "ret":
        # look for Ret_cellClassifications.csv (case-insensitive)
        cand = None
        for fn in os.listdir(sess_dir):
            if fn.lower() == "ret_cellclassifications.csv":
                cand = os.path.join(sess_dir, fn)
                break
        if cand and os.path.exists(cand):
            df = pd.read_csv(cand)
            if "group" in df.columns and "class" not in df.columns:
                df = df.rename(columns={"group":"class"})
            if "animal_id" not in df.columns or "cell_id" not in df.columns or "class" not in df.columns:
                raise ValueError("Ret_cellClassifications.csv missing required columns.")
            df["animal_id"] = df["animal_id"].astype(str).map(lambda x: norm_animal_id(x))
            df["cell_id"] = df["cell_id"].astype(str).map(lambda x: normalize_cell_id_any(x))
            df["class"] = df["class"].astype(str).str.strip()
            m = {
                "early_only":"EarlyOnly","late_only":"LateOnly","overlap":"Overlap","neither":"Neither",
                "earlyonly":"EarlyOnly","lateonly":"LateOnly"
            }
            df["class_norm"] = df["class"].str.replace(" ","").str.lower().map(lambda x: m.get(x, x))
            df["class_norm"] = df["class_norm"].map(lambda x: x if x in VALID else str(x))
            out = {}
            for aid, sub in df.groupby("animal_id"):
                out[aid] = sub[["cell_id","class_norm"]].rename(columns={"class_norm":"class"})
            return out, sess_dir
    files = sorted(glob.glob(os.path.join(sess_dir, f"*_{session_tag}_onset_evoked_cell_classes.csv")))
    if not files:
        # fallback: any csv with session tag
        files = sorted(glob.glob(os.path.join(sess_dir, f"*{session_tag}*cell*class*.csv")))
    out={}
    for fp in files:
        a = infer_animal_id(fp)
        df = pd.read_csv(fp)
        if "class" not in df.columns and "group" in df.columns:
            df = df.rename(columns={"group":"class"})
        if "cell_id" not in df.columns:
            for c in df.columns:
                if "cell" in c.lower():
                    df = df.rename(columns={c:"cell_id"})
                    break
        df = df[["cell_id","class"]].copy()
        df["cell_id"] = df["cell_id"].astype(str).map(lambda x: normalize_cell_id_any(x))
        df["class"] = df["class"].astype(str).str.strip()
        # normalize possible variants
        m = {
            "early_only":"EarlyOnly","late_only":"LateOnly","overlap":"Overlap","neither":"Neither",
            "earlyonly":"EarlyOnly","lateonly":"LateOnly"
        }
        df["class_norm"] = df["class"].str.replace(" ","").str.lower().map(lambda x: m.get(x, x))
        df["class_norm"] = df["class_norm"].map(lambda x: x if x in VALID else str(x))
        out[a] = df[["cell_id","class_norm"]].rename(columns={"class_norm":"class"})
    return out, sess_dir

def _to_int_id(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None

def _best_id_alignment(src_df: pd.DataFrame, tgt_df: pd.DataFrame):
    d1 = src_df.copy()
    d2 = tgt_df.copy()
    d1["_id_int"] = d1["cell_id"].map(_to_int_id)
    d2["_id_int"] = d2["cell_id"].map(_to_int_id)

    s1 = set(v for v in d1["_id_int"].dropna().astype(int).tolist())
    s2_raw = set(v for v in d2["_id_int"].dropna().astype(int).tolist())
    raw_overlap = len(s1.intersection(s2_raw))

    cands = [("identity", d2["_id_int"])]
    cands.append((
        "div10_when_divisible",
        d2["_id_int"].map(
            lambda v: (int(v) // 10)
            if pd.notna(v) and int(v) % 10 == 0
            else (int(v) if pd.notna(v) else np.nan)
        ),
    ))

    best_name, best_series, best_overlap = "identity", d2["_id_int"], raw_overlap
    for nm, ser in cands:
        s2 = set(v for v in ser.dropna().astype(int).tolist())
        ov = len(s1.intersection(s2))
        if ov > best_overlap:
            best_name, best_series, best_overlap = nm, ser, ov

    out = d2.copy()
    out["cell_id"] = best_series.map(lambda v: str(int(v)) if pd.notna(v) else None)
    out = out.dropna(subset=["cell_id"]).drop_duplicates(subset=["cell_id"], keep="first")
    return out[["cell_id", "class"]], best_name, raw_overlap, best_overlap

def compute_transitions(src_map: dict, tgt_map: dict):
    animals = sorted(set(src_map.keys()).intersection(tgt_map.keys()))
    rows = []
    mats = {}
    diagnostics = []
    for a in animals:
        d1 = src_map[a].rename(columns={"class": "src"})
        d2_src = tgt_map[a].rename(columns={"class": "tgt"})
        d2_aligned, strategy, raw_ov, best_ov = _best_id_alignment(
            d1.rename(columns={"src": "class"}), d2_src.rename(columns={"tgt": "class"})
        )
        d2 = d2_aligned.rename(columns={"class": "tgt"})
        m = d1.merge(d2, on="cell_id", how="inner")
        if m.empty:
            continue
        ct = pd.crosstab(m["src"], m["tgt"]).reindex(index=VALID, columns=VALID, fill_value=0)
        mats[a] = ct
        diagnostics.append({
            "animal_id": a,
            "sex": sex_of(a),
            "id_alignment_strategy": strategy,
            "raw_overlap": int(raw_ov),
            "best_overlap": int(best_ov),
            "n_src_cells": int(d1["cell_id"].nunique()),
            "n_tgt_cells": int(d2_src["cell_id"].nunique()),
            "n_registered": int(m["cell_id"].nunique()),
        })
        for src in VALID:
            for tgt in VALID:
                rows.append({"animal_id": a, "sex": sex_of(a), "src": src, "tgt": tgt, "count": int(ct.loc[src, tgt])})
    return pd.DataFrame(rows), mats, pd.DataFrame(diagnostics)

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not np.any(ok):
        return q
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    qv = np.empty_like(pv)
    qv[order] = adj
    q[ok] = qv
    return q

def safe_sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))

def perm_p_mean_diff(a, b, n_perm=10000, seed=0):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(a) - np.mean(b)))
    comb = np.concatenate([a, b])
    na = a.size
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(comb)
        d = abs(float(np.mean(comb[:na]) - np.mean(comb[na:])))
        if d >= obs - 1e-15:
            ge += 1
    return (ge + 1) / (n_perm + 1)

def two_prop_stats(success_f, total_f, success_m, total_m):
    if total_f <= 0 or total_m <= 0:
        return {
            "female_prop": np.nan, "male_prop": np.nan, "prop_difference_female_minus_male": np.nan,
            "prop_diff_95CI_low": np.nan, "prop_diff_95CI_high": np.nan, "z_stat_2prop": np.nan, "p_2prop_two_tailed": np.nan,
            "chi2_no_correction": np.nan, "p_chi2_no_correction": np.nan, "chi2_yates": np.nan, "p_chi2_yates": np.nan,
            "fisher_odds_ratio": np.nan, "p_fisher_two_tailed": np.nan,
        }
    pf = float(success_f) / float(total_f)
    pm = float(success_m) / float(total_m)
    diff = pf - pm
    se = np.sqrt(max(0.0, pf * (1.0 - pf) / total_f + pm * (1.0 - pm) / total_m))
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se

    p_pool = (success_f + success_m) / (total_f + total_m)
    se_pool = np.sqrt(max(0.0, p_pool * (1.0 - p_pool) * (1.0 / total_f + 1.0 / total_m)))
    if se_pool > 0:
        z = diff / se_pool
        pz = 2.0 * stats.norm.sf(abs(z))
    else:
        z, pz = np.nan, np.nan

    table = np.array([
        [success_f, max(0, total_f - success_f)],
        [success_m, max(0, total_m - success_m)],
    ], dtype=float)
    try:
        chi2_nc, pchi_nc, _, _ = stats.chi2_contingency(table, correction=False)
    except ValueError:
        chi2_nc, pchi_nc = np.nan, np.nan
    try:
        chi2_y, pchi_y, _, _ = stats.chi2_contingency(table, correction=True)
    except ValueError:
        chi2_y, pchi_y = np.nan, np.nan
    try:
        oratio, pfisher = stats.fisher_exact(table, alternative="two-sided")
    except ValueError:
        oratio, pfisher = np.nan, np.nan

    return {
        "female_prop": pf, "male_prop": pm, "prop_difference_female_minus_male": diff,
        "prop_diff_95CI_low": ci_low, "prop_diff_95CI_high": ci_high, "z_stat_2prop": z, "p_2prop_two_tailed": pz,
        "chi2_no_correction": chi2_nc, "p_chi2_no_correction": pchi_nc, "chi2_yates": chi2_y, "p_chi2_yates": pchi_y,
        "fisher_odds_ratio": oratio, "p_fisher_two_tailed": pfisher,
    }

def save_transition_tables(prefix: str, trans: pd.DataFrame, out_dir: str):
    counts = trans.rename(columns={"animal_id": "animal"})[["animal", "sex", "src", "tgt", "count"]].copy()
    counts = counts.sort_values(["animal", "src", "tgt"]).reset_index(drop=True)
    counts.to_csv(os.path.join(out_dir, f"{prefix}_transition_counts_byAnimal.csv"), index=False)
    counts.to_csv(os.path.join(out_dir, f"{prefix}_transition_counts_byAnimal_ALL.csv"), index=False)
    for sx in ["Female", "Male"]:
        counts[counts["sex"] == sx].to_csv(os.path.join(out_dir, f"{prefix}_transition_counts_byAnimal_{sx}.csv"), index=False)

    denom = counts.groupby(["animal", "src"], as_index=False)["count"].sum().rename(columns={"count": "src_total"})
    props = counts.merge(denom, on=["animal", "src"], how="left")
    props["prop"] = np.where(props["src_total"] > 0, props["count"] / props["src_total"], np.nan)
    props_all = props[["animal", "sex", "src", "tgt", "count", "prop"]].copy()
    props_all = props_all.sort_values(["animal", "src", "tgt"]).reset_index(drop=True)
    props_all.to_csv(os.path.join(out_dir, f"{prefix}_transition_props_byAnimal_ALL.csv"), index=False)
    for sx in ["Female", "Male"]:
        psub = props_all[props_all["sex"] == sx].copy()
        psub.to_csv(os.path.join(out_dir, f"{prefix}_transition_props_byAnimal_{sx}.csv"), index=False)
        summ = psub.groupby(["src", "tgt"], as_index=False).agg(
            mean=("prop", "mean"),
            count=("prop", "count"),
            sem=("prop", safe_sem),
        )
        summ.to_csv(os.path.join(out_dir, f"{prefix}_transition_props_byAnimal_{sx}_meanSEM.csv"), index=False)
    return counts, props_all

def save_transition_sexdiff_stats(prefix: str, props_all: pd.DataFrame, out_dir: str):
    rows = []
    for src in VALID:
        for tgt in VALID:
            sub = props_all[(props_all["src"] == src) & (props_all["tgt"] == tgt)]
            m = sub[sub["sex"] == "Male"]["prop"].dropna().values
            f = sub[sub["sex"] == "Female"]["prop"].dropna().values
            if len(m) >= 2 and len(f) >= 2:
                t_res = stats.ttest_ind(m, f, equal_var=False, nan_policy="omit")
                welch_t, welch_p = float(t_res.statistic), float(t_res.pvalue)
                perm_p = perm_p_mean_diff(m, f, n_perm=10000, seed=42)
            else:
                welch_t, welch_p, perm_p = np.nan, np.nan, np.nan
            rows.append({
                "src": src,
                "tgt": tgt,
                "male_mean": float(np.nanmean(m)) if len(m) else np.nan,
                "female_mean": float(np.nanmean(f)) if len(f) else np.nan,
                "diff_m_f": (float(np.nanmean(m)) - float(np.nanmean(f))) if len(m) and len(f) else np.nan,
                "welch_t": welch_t,
                "welch_p": welch_p,
                "perm_p": perm_p,
                "n_male": int(len(m)),
                "n_female": int(len(f)),
            })
    out = pd.DataFrame(rows)
    out["welch_q"] = bh_fdr(out["welch_p"].values)
    out["perm_q"] = bh_fdr(out["perm_p"].values)
    out.to_csv(os.path.join(out_dir, f"{prefix}_transition_sexDiff_stats.csv"), index=False)
    return out

def save_source_retention_outputs(prefix: str, counts: pd.DataFrame, out_dir: str, diag: pd.DataFrame | None = None):
    c = counts.copy()
    c["retained"] = (c["src"] == c["tgt"]).astype(int) * c["count"]
    per = c.groupby(["animal", "sex", "src"], as_index=False).agg(
        total=("count", "sum"),
        retained=("retained", "sum"),
    )
    per["lost"] = per["total"] - per["retained"]
    per["prop_retained"] = np.where(per["total"] > 0, per["retained"] / per["total"], np.nan)
    per["prop_lost"] = np.where(per["total"] > 0, per["lost"] / per["total"], np.nan)
    per.to_csv(os.path.join(out_dir, f"{prefix}_transition_retention_bySrc_byAnimal.csv"), index=False)

    bysex = per.groupby(["sex", "src"], as_index=False).agg(
        mean=("prop_retained", "mean"),
        count=("prop_retained", "count"),
        sem=("prop_retained", safe_sem),
    )
    bysex.to_csv(os.path.join(out_dir, f"{prefix}_transition_retention_bySrc_bySex_meanSEM.csv"), index=False)
    overall = per.groupby(["src"], as_index=False).agg(
        mean=("prop_retained", "mean"),
        count=("prop_retained", "count"),
        sem=("prop_retained", safe_sem),
    )
    overall.to_csv(os.path.join(out_dir, f"{prefix}_transition_retention_bySrc_overall_meanSEM.csv"), index=False)

    rows = []
    for src in VALID:
        for metric in ["prop_retained", "prop_lost"]:
            m = per[(per["src"] == src) & (per["sex"] == "Male")][metric].dropna().values
            f = per[(per["src"] == src) & (per["sex"] == "Female")][metric].dropna().values
            if len(m) >= 2 and len(f) >= 2:
                t_res = stats.ttest_ind(m, f, equal_var=False, nan_policy="omit")
                welch_t, welch_p = float(t_res.statistic), float(t_res.pvalue)
                perm_p = perm_p_mean_diff(m, f, n_perm=10000, seed=121)
            else:
                welch_t, welch_p, perm_p = np.nan, np.nan, np.nan
            rows.append({
                "src": src,
                "metric": metric,
                "male_mean": float(np.nanmean(m)) if len(m) else np.nan,
                "female_mean": float(np.nanmean(f)) if len(f) else np.nan,
                "diff_m_f": (float(np.nanmean(m)) - float(np.nanmean(f))) if len(m) and len(f) else np.nan,
                "welch_t": welch_t,
                "welch_p": welch_p,
                "perm_p": perm_p,
                "n_male": int(len(m)),
                "n_female": int(len(f)),
            })
    sexdiff = pd.DataFrame(rows)
    sexdiff["welch_q"] = bh_fdr(sexdiff["welch_p"].values)
    sexdiff["perm_q"] = bh_fdr(sexdiff["perm_p"].values)
    sexdiff.to_csv(os.path.join(out_dir, f"{prefix}_transition_retention_bySrc_sexDiff_stats.csv"), index=False)

    for src in ["EarlyOnly", "LateOnly"]:
        sub = per[per["src"] == src].copy()
        rename = {
            "EarlyOnly": {
                "total": "earlyonly_total_count",
                "retained": "retained_count_earlyonly",
                "lost": "lost_count_earlyonly",
                "prop_retained": "prop_retained_earlyonly",
                "prop_lost": "prop_lost_earlyonly",
            },
            "LateOnly": {
                "total": "lateonly_total_count",
                "retained": "retained_count_lateonly",
                "lost": "lost_count_lateonly",
                "prop_retained": "prop_retained_lateonly",
                "prop_lost": "prop_lost_lateonly",
            },
        }[src]
        sub = sub.rename(columns=rename)
        if diag is not None and not diag.empty:
            dd = diag.rename(columns={"animal_id": "animal"})[
                ["animal", "id_alignment_strategy", "raw_overlap", "best_overlap"]
            ].drop_duplicates("animal")
            sub = dd.merge(sub, on="animal", how="right")
        cols = ["animal", "sex"]
        if "id_alignment_strategy" in sub.columns:
            cols += ["id_alignment_strategy", "raw_overlap", "best_overlap"]
        cols += [rename["total"], rename["retained"], rename["lost"], rename["prop_retained"], rename["prop_lost"]]
        sub = sub[cols]
        sub.to_csv(os.path.join(out_dir, f"{prefix}_{src}_retention_byAnimal.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    srcs = ["EarlyOnly", "Overlap", "LateOnly"]
    xpos = np.arange(len(srcs))
    width = 0.34
    for i, sx in enumerate(["Female", "Male"]):
        vals_ret = [float(bysex[(bysex["sex"] == sx) & (bysex["src"] == s)]["mean"].iloc[0]) if not bysex[(bysex["sex"] == sx) & (bysex["src"] == s)].empty else np.nan for s in srcs]
        vals_lost = [1.0 - v if np.isfinite(v) else np.nan for v in vals_ret]
        xx = xpos + (i - 0.5) * width
        ax.bar(xx, vals_ret, width=width, label=f"{sx} retained")
        ax.bar(xx, vals_lost, width=width, bottom=vals_ret, label=f"{sx} lost", alpha=0.55)
    ax.set_xticks(xpos)
    ax.set_xticklabels(srcs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_retention_bySrc_stacked_retainedLost_bySex.png"), dpi=300)
    fig.savefig(os.path.join(out_dir, f"{prefix}_retention_bySrc_stacked_retainedLost_bySex.svg"))
    plt.close(fig)

def save_tone_responsive_retention(prefix: str, counts: pd.DataFrame, out_dir: str):
    responsive = {"EarlyOnly", "LateOnly", "Overlap"}
    rows = []
    for aid, sub in counts.groupby("animal"):
        sx = str(sub["sex"].iloc[0])
        src_total = int(sub[sub["src"].isin(responsive)]["count"].sum())
        tgt_retained = int(sub[(sub["src"].isin(responsive)) & (sub["tgt"].isin(responsive))]["count"].sum())
        tgt_non = src_total - tgt_retained
        pr = (tgt_retained / src_total) if src_total > 0 else np.nan
        pl = (tgt_non / src_total) if src_total > 0 else np.nan
        rows.append({
            "animal": aid,
            "sex": sx,
            "ext2_responsive_total": src_total,
            "ret_responsive_from_ext2_resp": tgt_retained,
            "ret_nonresponsive_from_ext2_resp": tgt_non,
            "prop_retained_resp": pr,
            "prop_lost_resp": pl,
        })
    out = pd.DataFrame(rows).sort_values("animal")
    out.to_csv(os.path.join(out_dir, f"{prefix}_transition_toneResponsive_byAnimal.csv"), index=False)

    for metric, fname in [
        ("prop_lost_resp", f"{prefix}_transition_sexDiff_toneResponsive_stats.csv"),
        ("prop_retained_resp", f"{prefix}_transition_sexDiff_toneResponsive_retention_stats.csv"),
    ]:
        m = out[out["sex"] == "Male"][metric].dropna().values
        f = out[out["sex"] == "Female"][metric].dropna().values
        if len(m) >= 2 and len(f) >= 2:
            t_res = stats.ttest_ind(m, f, equal_var=False, nan_policy="omit")
            row = {
                "metric": metric,
                "male_mean": float(np.mean(m)),
                "female_mean": float(np.mean(f)),
                "diff_m_f": float(np.mean(m) - np.mean(f)),
                "welch_t": float(t_res.statistic),
                "welch_p": float(t_res.pvalue),
                "n_male": int(len(m)),
                "n_female": int(len(f)),
            }
        else:
            row = {"metric": metric, "male_mean": np.nan, "female_mean": np.nan, "diff_m_f": np.nan, "welch_t": np.nan, "welch_p": np.nan, "n_male": len(m), "n_female": len(f)}
        pd.DataFrame([row]).to_csv(os.path.join(out_dir, fname), index=False)
    return out

def save_source_conditional_totalcount_sexcomp(prefix: str, counts: pd.DataFrame, out_dir: str):
    rows = []
    for src in ["LateOnly", "EarlyOnly"]:
        sub = counts[counts["src"] == src]
        f_total = int(sub[sub["sex"] == "Female"]["count"].sum())
        m_total = int(sub[sub["sex"] == "Male"]["count"].sum())
        for tgt in VALID:
            f_num = int(sub[(sub["sex"] == "Female") & (sub["tgt"] == tgt)]["count"].sum())
            m_num = int(sub[(sub["sex"] == "Male") & (sub["tgt"] == tgt)]["count"].sum())
            st = two_prop_stats(f_num, f_total, m_num, m_total)
            rows.append({
                "transition": f"{src}->{tgt}",
                "src": src,
                "tgt": tgt,
                "female_transition_cells": f_num,
                "female_source_total_cells": f_total,
                "male_transition_cells": m_num,
                "male_source_total_cells": m_total,
                "n_animals_female": int(counts[counts["sex"] == "Female"]["animal"].nunique()),
                "n_animals_male": int(counts[counts["sex"] == "Male"]["animal"].nunique()),
                **st,
            })
    out = pd.DataFrame(rows)
    out["p_2prop_two_tailed_fdr_bh"] = bh_fdr(out["p_2prop_two_tailed"].values)
    out["p_chi2_no_correction_fdr_bh"] = bh_fdr(out["p_chi2_no_correction"].values)
    out["p_fisher_two_tailed_fdr_bh"] = bh_fdr(out["p_fisher_two_tailed"].values)
    out.to_csv(os.path.join(out_dir, f"{prefix}_sourceConditional_totalcount_sexComparison_EarlyLateOnly.csv"), index=False)

def save_source_specific_outputs(prefix: str, counts: pd.DataFrame, props_all: pd.DataFrame, out_dir: str, left_label="Ext2", right_label="Ret"):
    for src in ["EarlyOnly", "LateOnly"]:
        for sex_name in ["All", "Female", "Male"]:
            if sex_name == "All":
                csub = counts[counts["src"] == src]
                psub = props_all[props_all["src"] == src]
            else:
                csub = counts[(counts["src"] == src) & (counts["sex"] == sex_name)]
                psub = props_all[(props_all["src"] == src) & (props_all["sex"] == sex_name)]
            if csub.empty:
                continue
            cc = csub.groupby(["src", "tgt"], as_index=False)["count"].sum()
            cc.to_csv(os.path.join(out_dir, f"{prefix}_transition_counts_{src}_{sex_name}.csv"), index=False)
            base = os.path.join(out_dir, f"{prefix}_alluvial_{src}_{sex_name}")
            alluvial_plot(
                cc,
                "count",
                f"{left_label} {src} transitions ({sex_name}, cell-weighted)",
                base,
                left_order=[src],
                right_order=VALID,
                left_label=left_label,
                right_label=right_label,
                show_pct_labels=True,
            )
            sankey_plot_html(cc, "count", f"{left_label} {src} transitions ({sex_name}, cell-weighted)", os.path.join(out_dir, f"{prefix}_Sankey_{src}_{sex_name}.html"), left_label=left_label, right_label=right_label)

            if psub.empty:
                continue
            agg = psub.groupby(["src", "tgt"], as_index=False).agg(
                pct=("prop", lambda x: 100.0 * float(np.nanmean(x))),
                n_animals=("prop", lambda x: int(np.isfinite(x).sum())),
            )
            agg.to_csv(os.path.join(out_dir, f"{prefix}_transition_props_animalWeighted_{src}_{sex_name}.csv"), index=False)
            base2 = os.path.join(out_dir, f"{prefix}_alluvial_{src}_{sex_name}_animalWeighted_meanPct")
            alluvial_plot(
                agg,
                "pct",
                f"{left_label} {src} transitions ({sex_name}, animal-weighted %)",
                base2,
                left_order=[src],
                right_order=VALID,
                left_label=left_label,
                right_label=right_label,
                show_pct_labels=True,
            )
            sankey_plot_html(agg, "pct", f"{left_label} {src} transitions ({sex_name}, animal-weighted %)", os.path.join(out_dir, f"{prefix}_Sankey_{src}_{sex_name}_animalWeighted_meanPct.html"), left_label=left_label, right_label=right_label)

def save_retrieval_denominator_retention(prefix: str, counts: pd.DataFrame, out_dir: str):
    responsive = {"EarlyOnly", "LateOnly", "Overlap"}
    rows = []
    for aid, sub in counts.groupby("animal"):
        sx = str(sub["sex"].iloc[0])
        ret_resp_total = int(sub[sub["tgt"].isin(responsive)]["count"].sum())
        retained = int(sub[(sub["src"].isin(responsive)) & (sub["tgt"].isin(responsive))]["count"].sum())
        prop = retained / ret_resp_total if ret_resp_total > 0 else np.nan
        rows.append({
            "animal_id": aid,
            "sex": sx,
            "retrieval_tone_responsive_total": ret_resp_total,
            "retained_resp_from_ext2_matched": retained,
            "prop_retained_of_retrieval_responsive": prop,
        })
    out = pd.DataFrame(rows).sort_values("animal_id")
    out.to_csv(os.path.join(out_dir, f"{prefix}_retrievalResponsiveRetention_byAnimal.csv"), index=False)

    agg = out.groupby("sex", as_index=False).agg(
        numerator_cells=("retained_resp_from_ext2_matched", "sum"),
        retrieval_responsive_total=("retrieval_tone_responsive_total", "sum"),
    )
    ext2_rows = []
    for sx, sub in counts.groupby("sex"):
        ext2_rows.append({
            "sex": sx,
            "ext2_responsive_total": int(sub[sub["src"].isin(responsive)]["count"].sum()),
        })
    ext2_agg = pd.DataFrame(ext2_rows)
    cmp_df = agg.merge(ext2_agg, on="sex", how="left")
    cmp_df["prop_of_retrieval_responsive"] = cmp_df["numerator_cells"] / cmp_df["retrieval_responsive_total"]
    cmp_df["prop_of_ext2_responsive"] = cmp_df["numerator_cells"] / cmp_df["ext2_responsive_total"]
    cmp_df.to_csv(os.path.join(out_dir, f"{prefix}_retrievalResponsiveRetention_denominatorComparison_bySex.csv"), index=False)

    f = cmp_df[cmp_df["sex"] == "Female"]
    m = cmp_df[cmp_df["sex"] == "Male"]
    if (not f.empty) and (not m.empty):
        st = two_prop_stats(
            int(f["numerator_cells"].iloc[0]), int(f["retrieval_responsive_total"].iloc[0]),
            int(m["numerator_cells"].iloc[0]), int(m["retrieval_responsive_total"].iloc[0]),
        )
        row = {
            "metric": "Proportion of Retrieval tone-responsive cells retained from Ext2",
            "numerator_definition": "matched cells with Ext2 in {EarlyOnly,LateOnly,Overlap} and Ret in {EarlyOnly,LateOnly,Overlap}",
            "denominator_definition": "all Retrieval tone-responsive cells (Ret class in {EarlyOnly,LateOnly,Overlap})",
            "female_numerator_cells": int(f["numerator_cells"].iloc[0]),
            "female_denominator_cells": int(f["retrieval_responsive_total"].iloc[0]),
            "male_numerator_cells": int(m["numerator_cells"].iloc[0]),
            "male_denominator_cells": int(m["retrieval_responsive_total"].iloc[0]),
            "n_animals_female": int(out[out["sex"] == "Female"]["animal_id"].nunique()),
            "n_animals_male": int(out[out["sex"] == "Male"]["animal_id"].nunique()),
            **st,
        }
        pd.DataFrame([row]).to_csv(os.path.join(out_dir, f"{prefix}_retrievalResponsiveRetention_sexComparison_totalcount.csv"), index=False)

    ff = out[out["sex"] == "Female"]["prop_retained_of_retrieval_responsive"].dropna().values
    mm = out[out["sex"] == "Male"]["prop_retained_of_retrieval_responsive"].dropna().values
    if len(ff) >= 2 and len(mm) >= 2:
        t_res = stats.ttest_ind(ff, mm, equal_var=False, nan_policy="omit")
        aw = pd.DataFrame([{
            "metric": "Per-animal proportion retained of retrieval tone-responsive cells",
            "female_mean": float(np.mean(ff)),
            "male_mean": float(np.mean(mm)),
            "diff_female_minus_male": float(np.mean(ff) - np.mean(mm)),
            "welch_t": float(t_res.statistic),
            "welch_p": float(t_res.pvalue),
            "n_female": int(len(ff)),
            "n_male": int(len(mm)),
        }])
        aw.to_csv(os.path.join(out_dir, f"{prefix}_retrievalResponsiveRetention_sexComparison_animalWeighted.csv"), index=False)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ret_zip", required=True, help="Retrieval classes zip (e.g., /mnt/data/ret_classes_proportions_only.zip)")
    ap.add_argument("--ext2_raw_zips", required=True, nargs="+", help="One or more Ext2 raw trace zips")
    ap.add_argument("--out_dir", default="/mnt/data")
    ap.add_argument("--work_dir", default="/mnt/data/_ext2_ret_transition_work")

    # classification parameters (match Ext2 rule)
    ap.add_argument("--dz_thresh", type=float, default=0.5)
    ap.add_argument("--consec_frames", type=int, default=10)
    ap.add_argument("--post_win_s", type=float, default=3.0)
    ap.add_argument("--early_n", type=int, default=3)
    ap.add_argument("--late_n", type=int, default=3)
    ap.add_argument("--min_hits_per_period", type=int, default=2)

    args=ap.parse_args()

    out_dir=ensure_dir(args.out_dir)
    work_dir=ensure_dir(args.work_dir)
    ext2_raw_dir=ensure_clean_dir(os.path.join(work_dir,"ext2_raw"))
    ext2_out_dir=ensure_clean_dir(os.path.join(work_dir,"ext2_classes"))

    # 1) Build Ext2 class tables from raw traces
    extract_all(args.ext2_raw_zips, ext2_raw_dir)
    csvs = sorted(glob.glob(os.path.join(ext2_raw_dir, "**", "*.csv"), recursive=True))
    if not csvs:
        raise RuntimeError(f"No CSVs found in extracted ext2_raw_dir: {ext2_raw_dir}")

    class_files=[]
    prop_rows=[]
    excluded=[]
    seen_animals=set()
    for fp in csvs:
        a = infer_animal_id(fp)
        if a in seen_animals:
            excluded.append({"file":os.path.basename(fp), "animal_id":a, "error":"Duplicate animal_id in extracted CSVs; skipped"})
            continue
        try:
            df = pd.read_csv(fp)
            cls = classify_one_animal(
                df, args.dz_thresh, args.consec_frames, args.post_win_s,
                args.early_n, args.late_n, args.min_hits_per_period
            )
            out_fp = os.path.join(ext2_out_dir, f"{a}_ext2_onset_evoked_cell_classes.csv")
            cls[["cell_id","class","early_peak_dZ","late_peak_dZ","dt","n_tones"]].to_csv(out_fp, index=False)
            class_files.append(out_fp)
            seen_animals.add(a)

            # proportions
            tot=len(cls)
            prop = cls["class"].value_counts().reindex(VALID, fill_value=0)/max(1,tot)
            prop_rows.append({"animal_id":a, **{f"prop_{k.lower()}":float(prop[k]) for k in VALID}, "n_cells":int(tot), "sex":sex_of(a)})
        except Exception as e:
            excluded.append({"file":os.path.basename(fp), "animal_id":a, "error":f"{type(e).__name__}: {e}"})

    if not class_files:
        raise RuntimeError("Failed to create any Ext2 class files.")

    prop_df=pd.DataFrame(prop_rows).sort_values("animal_id")
    prop_csv=os.path.join(ext2_out_dir,"ext2_onset_evoked_proportions_by_animal.csv")
    prop_df.to_csv(prop_csv, index=False)

    # zip ext2 outputs (small)
    ext2_zip_out=os.path.join(out_dir,"ext2_classes_proportions_only.zip")
    with zipfile.ZipFile(ext2_zip_out,"w",zipfile.ZIP_DEFLATED) as zf:
        for fp in class_files:
            zf.write(fp, arcname=os.path.basename(fp))
        zf.write(prop_csv, arcname=os.path.basename(prop_csv))
        readme = "Ext2 onset-evoked classes generated from raw traces using Ext2 rule (ΔZ>=dz_thresh for consec_frames within 0..post_win_s)."
        zf.writestr("README_ext2_processing.txt", readme)

    # 2) Load Retrieval from provided zip + Ext2 from newly created zip
    ret, _ = load_class_tables_from_zip(args.ret_zip, "ret", work_dir)
    ext2, _ = load_class_tables_from_zip(ext2_zip_out, "ext2", work_dir)

    trans, mats, diag = compute_transitions(ext2, ret)
    if trans.empty:
        raise RuntimeError("No registered cells found after merging Ext2 and Retrieval by cell_id.")

    prefix = "Ext2toRet"

    counts_by_animal, props_by_animal = save_transition_tables(prefix, trans, out_dir)
    save_transition_sexdiff_stats(prefix, props_by_animal, out_dir)
    save_source_retention_outputs(prefix, counts_by_animal, out_dir, diag=diag)
    save_tone_responsive_retention(prefix, counts_by_animal, out_dir)
    save_source_conditional_totalcount_sexcomp(prefix, counts_by_animal, out_dir)
    save_source_specific_outputs(prefix, counts_by_animal, props_by_animal, out_dir, left_label="Ext2", right_label="Ret")
    save_retrieval_denominator_retention(prefix, counts_by_animal, out_dir)

    # Core cell-weighted and animal-weighted tables (backward-compatible names)
    cell_counts = trans.groupby(["src", "tgt"], as_index=False)["count"].sum()
    counts_csv = os.path.join(out_dir, f"{prefix}_transition_counts_cellWeighted.csv")
    cell_counts.to_csv(counts_csv, index=False)

    prop_rows = []
    for a, ct in mats.items():
        tot = ct.values.sum()
        if tot <= 0:
            continue
        ct_reset = (ct / tot).reset_index()
        if "index" in ct_reset.columns:
            ct_reset = ct_reset.rename(columns={"index": "src"})
        elif "src" not in ct_reset.columns:
            ct_reset = ct_reset.rename(columns={ct_reset.columns[0]: "src"})
        p = ct_reset.melt(id_vars="src", var_name="tgt", value_name="prop")
        p["animal_id"] = a
        p["sex"] = sex_of(a)
        prop_rows.append(p)
    props = pd.concat(prop_rows, ignore_index=True)
    animal_props = props.groupby(["src", "tgt"], as_index=False)["prop"].mean()
    props_csv = os.path.join(out_dir, f"{prefix}_transition_props_animalWeighted.csv")
    animal_props.to_csv(props_csv, index=False)

    bysex_counts = {}
    bysex_props = {}
    for s in ["Female", "Male"]:
        sub = trans[trans["sex"] == s]
        if not sub.empty:
            bysex_counts[s] = sub.groupby(["src", "tgt"], as_index=False)["count"].sum()
            bysex_counts[s].to_csv(os.path.join(out_dir, f"{prefix}_transition_counts_cellWeighted_{s}.csv"), index=False)
        psub = props[props["sex"] == s]
        if not psub.empty:
            bysex_props[s] = psub.groupby(["src", "tgt"], as_index=False)["prop"].mean()
            bysex_props[s].to_csv(os.path.join(out_dir, f"{prefix}_transition_props_animalWeighted_{s}.csv"), index=False)

    # Core plots + sex-stratified plots
    alluvial_plot(cell_counts, "count", "Ext2 → Ret transitions (cell-weighted counts)", os.path.join(out_dir, f"{prefix}_alluvial_cellWeighted"), left_label="Ext2", right_label="Ret")
    ap = animal_props.copy()
    ap["pct"] = 100 * ap["prop"]
    alluvial_plot(ap, "pct", "Ext2 → Ret transitions (animal-weighted mean %)", os.path.join(out_dir, f"{prefix}_alluvial_animalWeighted_meanPct"), left_label="Ext2", right_label="Ret")
    for s in ["Female", "Male"]:
        if s in bysex_counts:
            alluvial_plot(bysex_counts[s], "count", f"Ext2 → Ret transitions (cell-weighted) — {s}", os.path.join(out_dir, f"{prefix}_alluvial_cellWeighted_{s}"), left_label="Ext2", right_label="Ret")
        if s in bysex_props:
            bp = bysex_props[s].copy()
            bp["pct"] = 100 * bp["prop"]
            alluvial_plot(bp, "pct", f"Ext2 → Ret transitions (animal-weighted mean %) — {s}", os.path.join(out_dir, f"{prefix}_alluvial_animalWeighted_meanPct_{s}"), left_label="Ext2", right_label="Ret")

    # 4) metadata + bundle
    meta = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "ret_zip": args.ret_zip,
        "ext2_raw_zips": args.ext2_raw_zips,
        "ext2_zip_out": ext2_zip_out,
        "params": {
            "dz_thresh": args.dz_thresh,
            "consec_frames": args.consec_frames,
            "post_win_s": args.post_win_s,
            "early_n": args.early_n,
            "late_n": args.late_n,
            "min_hits_per_period": args.min_hits_per_period
        },
        "excluded_ext2_files": excluded,
        "n_animals_overlap": int(len(set(ret.keys()).intersection(ext2.keys()))),
        "unknown_sex_animals": sorted([a for a in set(ret.keys()).intersection(ext2.keys()) if sex_of(a)=="Unknown"]),
        "sex_mapping": {"Female": sorted(list(FEMALE)), "Male": sorted(list(MALE))},
        "id_alignment_summary": diag.to_dict(orient="records"),
    }
    meta_path = os.path.join(out_dir, f"{prefix}_transition_run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if not diag.empty:
        diag.to_csv(os.path.join(out_dir, f"{prefix}_overlap_diagnostics_idAlignment.csv"), index=False)

    bundle = os.path.join(out_dir, "ext2_to_ret_transitions_alluvial_bundle.zip")
    bundle_files = [ext2_zip_out]
    for fn in os.listdir(out_dir):
        if fn.startswith("Ext2toRet_") and not fn.endswith(".zip"):
            bundle_files.append(os.path.join(out_dir, fn))
    bundle_files.extend([counts_csv, props_csv, meta_path])
    bundle_files = sorted(set(bundle_files))

    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in bundle_files:
            if os.path.exists(fp):
                zf.write(fp, arcname=os.path.basename(fp))

    print("Wrote bundle:", bundle)
    print("Also wrote:", ext2_zip_out)

if __name__ == "__main__":
    main()
