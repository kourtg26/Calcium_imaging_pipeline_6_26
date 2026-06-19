import os, re, glob, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/mnt/data"

RAW_ZIPS = [
    os.path.join(DATA_DIR, "ext1_raw_trace_files_part1.zip"),
    os.path.join(DATA_DIR, "ext1_raw_trace_files_part2.zip"),
    os.path.join(DATA_DIR, "ext1_raw_trace_files_part3.zip"),
    os.path.join(DATA_DIR, "ext1_raw_trace_files_part4.zip"),
]

WORK_DIR = os.path.join(DATA_DIR, "_ext1_build_work")
os.makedirs(WORK_DIR, exist_ok=True)

# Ext2 onset-evoked rule (your specs)
DZ_THRESH = 0.5
CONSEC_FRAMES = 10
POST_WIN_S = 3.0

EARLY_N = 3
LATE_N = 3

# Heatmap: tone-only segments concatenated
# We will use the minimum tone length in frames across all animals/tones to standardize.
# (In your Ext1 data this ends up being 200 frames per tone, and 12 tones.)
# -----------------------------


# -----------------------------
# HELPERS: parsing
# -----------------------------
def infer_animal_id(filename: str) -> str:
    # grabs leading token before first '_' if it looks like an animal id
    base = os.path.basename(filename)
    m = re.match(r"^([A-Za-z]*\d+)", base)
    if m:
        return m.group(1)
    return base.split("_")[0]

def find_time_col(cols):
    # prioritize common exact names
    exact = ["Time_s", "Time (s)", "Time(s)", "Time"]
    for c in exact:
        if c in cols:
            return c
    # otherwise any column containing "time"
    for c in cols:
        if "time" in c.lower():
            return c
    return None

def find_tone_col(df: pd.DataFrame):
    cols = list(df.columns)
    priority = ["ToneFlag", "WithinTone", "Within_Tone", "InTone", "In_Tone", "is_tone", "CS", "InTone"]
    for c in priority:
        if c in cols:
            return c
    # heuristic: column containing "tone" and mostly binary
    for c in cols:
        lc = c.lower()
        if "tone" in lc or "intone" in lc:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() > 0.5:
                v = s.dropna().values
                if np.isin(v, [0, 1]).mean() > 0.95:
                    return c
    # fallback: "CS" if binary
    if "CS" in cols:
        s = pd.to_numeric(df["CS"], errors="coerce")
        if s.notna().mean() > 0.5:
            v = s.dropna().values
            if np.isin(v, [0, 1]).mean() > 0.95:
                return "CS"
    return None

def find_freeze_col(df: pd.DataFrame):
    cols = list(df.columns)
    priority = ["FreezeFlag", "Freezing", "freezing", "Freezing_Index", "FreezingIndex"]
    for c in priority:
        if c in cols:
            return c
    for c in cols:
        if "freez" in c.lower():
            return c
    return None

def select_trace_cols(df: pd.DataFrame, time_col: str, tone_col: str, freeze_col: str | None):
    cols = list(df.columns)
    exclude = set([time_col, tone_col])
    if freeze_col is not None:
        exclude.add(freeze_col)

    # also exclude anything that looks like helper columns (time-rel, tone id, etc.)
    def is_aux(c):
        lc = c.lower()
        if "tone" in lc and c != tone_col:
            return True
        if "freez" in lc and freeze_col is not None and c != freeze_col:
            return True
        if "time" in lc and c != time_col:
            return True
        if "cs" == lc and c != tone_col:
            return True
        return False

    # first pass: columns named like C###
    trace_cols = []
    for c in cols:
        if c in exclude or is_aux(c):
            continue
        name = str(c).strip()
        if re.fullmatch(r"C\d+", name):
            trace_cols.append(c)

    if trace_cols:
        return trace_cols

    # fallback: numeric columns that are mostly finite and not aux
    numeric_cols = []
    for c in cols:
        if c in exclude or is_aux(c):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.95 and np.nanstd(s.values) > 1e-12:
            numeric_cols.append(c)

    return numeric_cols

def tone_bouts_from_flag(flag_01: np.ndarray):
    f = np.asarray(flag_01).astype(int)
    on = np.where((f[1:] == 1) & (f[:-1] == 0))[0] + 1
    off = np.where((f[1:] == 0) & (f[:-1] == 1))[0] + 1
    if f[0] == 1:
        on = np.r_[0, on]
    if f[-1] == 1:
        off = np.r_[off, len(f)]
    bouts = []
    for s, e in zip(on, off):
        if e > s:
            bouts.append((s, e))
    return bouts

def zscore_cols(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd

def has_run_ge(x: np.ndarray, thresh: float, run_len: int) -> bool:
    # x is 1D
    ok = (x >= thresh).astype(np.int32)
    if ok.sum() < run_len:
        return False
    # run-length detection
    run = 0
    for v in ok:
        if v:
            run += 1
            if run >= run_len:
                return True
        else:
            run = 0
    return False


# -----------------------------
# STEP 1: Extract all CSVs
# -----------------------------
for zp in RAW_ZIPS:
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(WORK_DIR)

csv_paths = sorted(glob.glob(os.path.join(WORK_DIR, "*.csv")))

# Some zips may contain alternate versions (e.g., a file without tone col).
# We'll load only files that have a tone flag column.
records = []  # per animal: dict with Z, time, toneflag, freezeflag, trace_cols, dt, bouts

for fp in csv_paths:
    df = pd.read_csv(fp)
    time_col = find_time_col(df.columns)
    if time_col is None:
        continue

    tone_col = find_tone_col(df)
    if tone_col is None:
        continue

    freeze_col = find_freeze_col(df)
    trace_cols = select_trace_cols(df, time_col, tone_col, freeze_col)

    if len(trace_cols) == 0:
        continue

    t = pd.to_numeric(df[time_col], errors="coerce").values.astype(np.float64)
    tone = pd.to_numeric(df[tone_col], errors="coerce").fillna(0).values.astype(int)
    freeze = None
    if freeze_col is not None:
        freeze = pd.to_numeric(df[freeze_col], errors="coerce").fillna(0).values.astype(int)

    X = df[trace_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    # assign clean cell IDs
    clean_ids = []
    for j, c in enumerate(trace_cols):
        name = str(c).strip()
        if re.fullmatch(r"C\d+", name):
            clean_ids.append(name)
        else:
            clean_ids.append(f"C{j:03d}")

    # compute dt
    tt = t[np.isfinite(t)]
    if len(tt) < 3:
        continue
    dt = float(np.median(np.diff(tt)))

    bouts = tone_bouts_from_flag(tone)
    if len(bouts) < (EARLY_N + LATE_N):
        continue

    animal_id = infer_animal_id(fp)
    records.append({
        "animal_id": animal_id,
        "file": fp,
        "t": t,
        "dt": dt,
        "tone": tone,
        "freeze": freeze,
        "X": X,
        "cell_ids": clean_ids,
        "bouts": bouts,
    })

if not records:
    raise RuntimeError("No valid Ext1 CSVs found with tone flags and trace columns.")

# enforce consistent number of tones across animals
n_tones = min(len(r["bouts"]) for r in records)
# compute min tone length across all animals/tones (frames)
tone_len_frames = min((e - s) for r in records for (s, e) in r["bouts"][:n_tones])

# -----------------------------
# STEP 2: Z-score per cell across full session, then classify by Ext2 ΔZ rule
# -----------------------------
all_rows = []        # for cell_order.csv
heatmap_rows = []    # per-cell concatenated tone-only vector

for r in records:
    animal = r["animal_id"]
    t = r["t"]
    dt = r["dt"]
    tone = r["tone"]
    X = r["X"]
    cell_ids = r["cell_ids"]
    bouts = r["bouts"][:n_tones]

    Z = zscore_cols(X)  # shape: T x Ncells

    # how many frames correspond to 0..POST_WIN_S?
    post_frames = int(np.floor(POST_WIN_S / dt))
    post_frames = max(1, post_frames)

    # define early and late tone indices
    early_idx = list(range(min(EARLY_N, n_tones)))
    late_idx = list(range(max(0, n_tones - LATE_N), n_tones))

    # precompute onset indices for each tone
    onsets = [s for (s, e) in bouts]

    # build tone-only concatenated Z for heatmap (exclude "Neither" later)
    # we first compute classification for each cell
    for j, cid in enumerate(cell_ids):
        # per-tone detection in 0..3s window using delta from onset sample
        early_hits = []
        late_hits = []

        # peaks used for sorting
        early_peak = -np.inf
        late_peak = -np.inf

        # Evaluate early tones
        for k in early_idx:
            s, e = bouts[k]
            idx0 = s  # onset sample index
            w_end = min(Z.shape[0], idx0 + post_frames + 1)
            dz = Z[idx0:w_end, j] - Z[idx0, j]
            if dz.size:
                early_peak = max(early_peak, float(np.nanmax(dz)))
                early_hits.append(has_run_ge(dz, DZ_THRESH, CONSEC_FRAMES))
            else:
                early_hits.append(False)

        # Evaluate late tones
        for k in late_idx:
            s, e = bouts[k]
            idx0 = s
            w_end = min(Z.shape[0], idx0 + post_frames + 1)
            dz = Z[idx0:w_end, j] - Z[idx0, j]
            if dz.size:
                late_peak = max(late_peak, float(np.nanmax(dz)))
                late_hits.append(has_run_ge(dz, DZ_THRESH, CONSEC_FRAMES))
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

        # store row for ordering
        all_rows.append({
            "animal_id": animal,
            "cell_id": cid,
            "group": group,
            "early_peak_dZ": early_peak if np.isfinite(early_peak) else np.nan,
            "late_peak_dZ": late_peak if np.isfinite(late_peak) else np.nan,
            "sort_score": sort_score if np.isfinite(sort_score) else np.nan,
        })

        # build tone-only concatenated vector for heatmap (Z during tone only)
        # tone segment: first tone_len_frames of each tone bout
        segs = []
        for k in range(n_tones):
            s, e = bouts[k]
            seg = Z[s:s + tone_len_frames, j]
            # seg length should match tone_len_frames by construction
            if seg.shape[0] != tone_len_frames:
                # if any oddity occurs, pad with NaN (should be rare if min length used)
                pad = tone_len_frames - seg.shape[0]
                if pad > 0:
                    seg = np.concatenate([seg, np.full(pad, np.nan, dtype=np.float32)])
                else:
                    seg = seg[:tone_len_frames]
            segs.append(seg.astype(np.float32))

        concat = np.concatenate(segs, axis=0)  # length = n_tones * tone_len_frames
        heatmap_rows.append(concat)

# Convert to DataFrames / arrays
meta = pd.DataFrame(all_rows)
H_all = np.vstack(heatmap_rows).astype(np.float32)  # rows correspond to meta rows

# Exclude "Neither" from the heatmap (to match your Ext2 scheme)
keep = meta["group"].isin(["EarlyOnly", "Overlap", "LateOnly"]).values
meta = meta.loc[keep].reset_index(drop=True)
H = H_all[keep, :]

# Sort: EarlyOnly (desc), Overlap (desc), LateOnly (desc)
group_order = ["EarlyOnly", "Overlap", "LateOnly"]
parts = []
for g in group_order:
    sub = meta[meta["group"] == g].copy()
    # default to descending sort_score; tie-break by animal_id then cell_id for stability
    sub = sub.sort_values(["sort_score", "animal_id", "cell_id"], ascending=[False, True, True])
    parts.append(sub)

meta_sorted = pd.concat(parts, ignore_index=True)

# reorder heatmap rows to match meta_sorted
# build index map from (animal_id, cell_id, group, sort_score) to row indices
# easiest: use original row positions tracked before filtering via a temporary index column
meta_keep = pd.DataFrame(all_rows).loc[keep].reset_index(drop=True)
meta_keep["_row"] = np.arange(len(meta_keep))
meta_sorted = meta_sorted.merge(
    meta_keep[["animal_id", "cell_id", "group", "sort_score", "_row"]],
    on=["animal_id", "cell_id", "group", "sort_score"],
    how="left"
)

row_idx = meta_sorted["_row"].values.astype(int)
H_sorted = H[row_idx, :]

# boundaries at end of each tone segment
boundaries = np.array([(i + 1) * tone_len_frames for i in range(n_tones)], dtype=int)

# -----------------------------
# STEP 3: Plot with group-specific gradients (clip -1..1), no labels
# -----------------------------
vmin, vmax = -1.0, 1.0
Hc = np.clip(H_sorted, vmin, vmax)
norm = (Hc - vmin) / (vmax - vmin)

base = {
    "EarlyOnly": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "Overlap":   np.array([0.5, 0.5, 0.5], dtype=np.float32),
    "LateOnly":  np.array([0.0, 0.0, 1.0], dtype=np.float32),
}
white = np.array([1.0, 1.0, 1.0], dtype=np.float32)

N, T = H_sorted.shape
rgb = np.empty((N, T, 3), dtype=np.float32)
groups = meta_sorted["group"].astype(str).tolist()
for i, g in enumerate(groups):
    b = base.get(g, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    a = norm[i, :].reshape(-1, 1)
    rgb[i, :, :] = white * (1 - a) + b * a

fig_h = min(18, max(5, N / 140))
fig_w = 15
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.imshow(rgb, aspect="auto", interpolation="nearest")

for b in boundaries[1:-1]:
    ax.axvline(b - 0.5, linewidth=0.6, color="k", alpha=0.25)

group_sizes = [int((meta_sorted["group"] == g).sum()) for g in group_order]
y = 0
for sz in group_sizes[:-1]:
    y += sz
    ax.axhline(y - 0.5, linewidth=1.0, color="k", alpha=0.35)

ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

out_png = os.path.join(
    DATA_DIR,
    "Ext1_toneOnly_heatmap_groupSpecificGradients_clip-1to1_NO_LABELS.png"
)
plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
plt.close()

# -----------------------------
# STEP 4: Save NPZ + order CSV + bundle
# -----------------------------
npz_out = os.path.join(DATA_DIR, "Ext1_toneOnly_heatmap_matrix.npz")
csv_out = os.path.join(DATA_DIR, "Ext1_toneOnly_heatmap_cell_order.csv")

np.savez_compressed(npz_out, heatmap=H_sorted.astype(np.float32), boundaries=boundaries.astype(int))
meta_sorted.drop(columns=["_row"], errors="ignore").to_csv(csv_out, index=False)

bundle_out = os.path.join(DATA_DIR, "ext1_toneOnly_heatmap_earlyTop_lateBottom_bundle.zip")
with zipfile.ZipFile(bundle_out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(npz_out, arcname=os.path.basename(npz_out))
    zf.write(csv_out, arcname=os.path.basename(csv_out))
    zf.write(out_png, arcname=os.path.basename(out_png))

print("Wrote:")
print(" ", out_png)
print(" ", npz_out)
print(" ", csv_out)
print(" ", bundle_out)
print(f"Tone count = {n_tones}, tone_len_frames = {tone_len_frames}, heatmap shape = {H_sorted.shape}")
