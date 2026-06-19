import os, re, zipfile, json
import numpy as np
import pandas as pd

# -----------------------
# Generic helpers
# -----------------------
def infer_fps(t: np.ndarray) -> float:
    t=np.asarray(t, float)
    dt=np.diff(t)
    dt=dt[np.isfinite(dt) & (dt>0)]
    if len(dt)==0:
        return float("nan")
    med=np.median(dt)
    return float(1.0/med) if med>0 else float("nan")

def rising_edges(flag: np.ndarray) -> np.ndarray:
    f=np.nan_to_num(np.asarray(flag), nan=0.0).astype(int)
    if len(f)==0:
        return np.array([], dtype=int)
    edges=np.where((f[1:]==1) & (f[:-1]==0))[0] + 1
    if f[0]==1:
        edges=np.r_[0, edges]
    return edges.astype(int)

def standardize_class_label(x) -> str:
    x=str(x).strip().lower().replace(" ", "_")
    if x in ("early_only","earlyonly"): return "EarlyOnly"
    if x in ("late_only","lateonly"): return "LateOnly"
    if x in ("overlap",): return "Overlap"
    if x in ("neither","none","na","nan",""): return "Neither"
    if x in ("earlyonly","lateonly","overlap","neither"):
        return x[0].upper()+x[1:]
    return x

def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+","", str(c).strip().lower())

def pick_col(cols, candidates):
    m={norm_col(c): c for c in cols}
    for cand in candidates:
        k=norm_col(cand)
        if k in m:
            return m[k]
    for cand in candidates:
        k=norm_col(cand)
        for kk, orig in m.items():
            if k and k in kk:
                return orig
    return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

# -----------------------
# Sex mapping
# -----------------------
def sex_from_animal_id(animal_id: str, sex_mapping: dict) -> str:
    aid=str(animal_id).strip()
    for sex, ids in sex_mapping.items():
        if aid in [str(x).strip() for x in ids]:
            return sex
    return "Unknown"

# -----------------------
# Load class maps (Ext1, Retrieval)
# -----------------------
def load_ext1_class_maps(ext1_classes_zip: str, workdir: str) -> dict:
    """Returns dict[animal_id] -> dict[cell_id] -> class."""
    ensure_dir(workdir)
    with zipfile.ZipFile(ext1_classes_zip, "r") as zf:
        zf.extractall(workdir)
    out={}
    for fp in [os.path.join(workdir,f) for f in os.listdir(workdir) if f.endswith("_ext1_onset_evoked_cell_classes.csv")]:
        aid=os.path.basename(fp).split("_ext1_onset_evoked_cell_classes.csv")[0]
        d=pd.read_csv(fp)
        d["cell_id"]=d["cell_id"].astype(str).str.strip()
        d["class"]=d["class"].astype(str).map(standardize_class_label)
        out[aid]=d.set_index("cell_id")["class"].to_dict()
    return out

def load_ret_class_maps(ret_classes_zip: str, workdir: str) -> dict:
    """Returns dict[animal_id] -> dict[cell_id] -> class."""
    ensure_dir(workdir)
    with zipfile.ZipFile(ret_classes_zip, "r") as zf:
        zf.extractall(workdir)
    fp=os.path.join(workdir, "Ret_cellClassifications.csv")
    if not os.path.exists(fp):
        # try any CSV that looks like cell classifications
        cands=[f for f in os.listdir(workdir) if f.lower().endswith(".csv")]
        if len(cands)==0:
            return {}
        fp=os.path.join(workdir, cands[0])
    d=pd.read_csv(fp)
    d["animal_id"]=d["animal_id"].astype(str).str.strip()
    d["cell_id"]=d["cell_id"].astype(str).str.strip()
    grp_col="group" if "group" in d.columns else ("class" if "class" in d.columns else None)
    if grp_col is None:
        # pick last column as fallback
        grp_col=d.columns[-1]
    d[grp_col]=d[grp_col].astype(str).map(standardize_class_label)
    out={}
    for aid, sub in d.groupby("animal_id"):
        out[str(aid).strip()]=sub.set_index("cell_id")[grp_col].to_dict()
    return out

# -----------------------
# Load zscored NPZs (Ext1 parts, Retrieval bundle)
# -----------------------
def extract_npz_from_zip(zp: str, outdir: str) -> list:
    ensure_dir(outdir)
    with zipfile.ZipFile(zp, "r") as zf:
        for n in zf.namelist():
            if n.lower().endswith(".npz"):
                out=os.path.join(outdir, os.path.basename(n))
                if not os.path.exists(out):
                    zf.extract(n, path=outdir)
    return sorted([os.path.join(outdir,f) for f in os.listdir(outdir) if f.lower().endswith(".npz")])

def extract_npz_parts(zips: list, outdir: str) -> list:
    ensure_dir(outdir)
    paths=[]
    for zp in zips:
        if not os.path.exists(zp):
            continue
        paths += extract_npz_from_zip(zp, outdir)
    # unique
    return sorted(list(dict.fromkeys(paths)))

def load_ext1_npz(npz_path: str) -> dict:
    npz=np.load(npz_path, allow_pickle=True)
    return {
        "time": npz["Time_s"].astype(float),
        "z": npz["Z"].astype(float),
        "tone_flag": npz["ToneFlag"].astype(int),
        "freeze": npz["FreezeFlag"].astype(int),
        "tone_id": npz["ToneIndex"].astype(int),
        "cell_ids": np.array([str(x).strip() for x in npz["cell_ids"]], dtype=str),
    }

def load_ret_npz(npz_path: str) -> dict:
    npz=np.load(npz_path, allow_pickle=True)
    return {
        "time": npz["time"].astype(float),
        "z": npz["z"].astype(float),
        "tone_flag": npz["tone_flag"].astype(int),
        "freeze": npz["freeze"].astype(int),
        "tone_id": npz["tone_id"].astype(int),
        "cell_ids": np.array([str(x).strip() for x in npz["cell_ids"]], dtype=str),
    }

# -----------------------
# Load Ext2 raw CSV + z-score
# -----------------------
def extract_csvs_from_zip(zp: str, outdir: str) -> list:
    ensure_dir(outdir)
    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(outdir)
    import glob
    return sorted(glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True))

def load_ext2_session_csv(fp: str) -> dict | None:
    df=pd.read_csv(fp)
    cols=list(df.columns)
    time_col=pick_col(cols, ["Time_s","Time (s)","Time(s)","time"])
    tone_flag_col=pick_col(cols, ["ToneFlag","WithinTone","InTone","is_tone","Within_Tone"])
    freeze_col=pick_col(cols, ["FreezeFlag","Freezing","freezing","Freeze"])
    tone_id_col=pick_col(cols, ["CS","ToneIndex","tone_id","ToneID","toneid"])

    if time_col is None or tone_flag_col is None or freeze_col is None:
        return None

    meta=set([time_col, tone_flag_col, freeze_col])
    if tone_id_col: meta.add(tone_id_col)

    cell_cols=[]
    for c in cols:
        if c in meta: 
            continue
        s=str(c).strip()
        if re.match(r"^C\d+$", s, flags=re.IGNORECASE) or re.match(r"^C\d{2,5}$", s, flags=re.IGNORECASE):
            cell_cols.append(c)
        elif s.lower().startswith("undecided"):
            cell_cols.append(c)
    if len(cell_cols)==0:
        # fallback: numeric columns that aren't meta
        for c in cols:
            if c in meta:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cell_cols.append(c)
    if len(cell_cols)==0:
        return None

    X=df[cell_cols].astype(float).values
    mu=np.nanmean(X, axis=0)
    sd=np.nanstd(X, axis=0, ddof=0)
    sd[(sd==0)|(~np.isfinite(sd))]=np.nan
    Z=(X-mu)/sd

    tone_flag=df[tone_flag_col].fillna(0).astype(int).values
    freeze=df[freeze_col].fillna(0).astype(int).values
    if tone_id_col:
        tone_id=df[tone_id_col].fillna(0).astype(int).values
    else:
        tone_id=tone_flag.copy()

    return {
        "time": df[time_col].astype(float).values,
        "z": Z.astype(float),
        "tone_flag": tone_flag.astype(int),
        "freeze": freeze.astype(int),
        "tone_id": tone_id.astype(int),
        "cell_ids": np.array([str(c).strip() for c in cell_cols], dtype=str),
    }

def choose_one_ext2_csv_per_animal(csv_paths: list) -> list:
    """Heuristic: choose the largest CSV per animal_id based on line count."""
    best={}
    for fp in csv_paths:
        base=os.path.basename(fp)
        m=re.match(r"(.+?)_ext2_", base)
        aid=m.group(1) if m else base.split("_")[0]
        try:
            nlines=sum(1 for _ in open(fp,'rb'))
        except Exception:
            nlines=0
        if aid not in best or nlines > best[aid][0]:
            best[aid]=(nlines, fp)
    return [v[1] for v in best.values()]

# -----------------------
# Tone-onset classification (Ext2; also usable elsewhere)
# -----------------------
def classify_onset_evoked(z: np.ndarray, tone_flag: np.ndarray, tone_id: np.ndarray, time: np.ndarray,
                          thr_z: float=0.5, consec_frames: int=10, window_s: float=3.0,
                          early_tones=(1,2,3), late_tones=(10,11,12)) -> np.ndarray:
    """Returns per-cell class: EarlyOnly / Overlap / LateOnly / Neither."""
    fps=infer_fps(time)
    if not np.isfinite(fps): fps=10.0
    win=int(round(window_s*fps))

    tone_onsets=rising_edges(tone_flag)
    nT, nC=z.shape
    early=np.zeros(nC, dtype=int)
    late=np.zeros(nC, dtype=int)

    for onset in tone_onsets:
        if onset>=nT:
            continue
        tid=int(tone_id[onset]) if onset < len(tone_id) else 0
        end=min(nT, onset+win)

        # onset-evoked increase relative to value at onset
        seg=z[onset:end, :] - z[onset:onset+1, :]
        above=(seg >= thr_z).astype(int)
        if above.shape[0] < consec_frames:
            continue
        cs=np.cumsum(above, axis=0)
        wsum=cs[consec_frames-1:, :] - np.vstack([np.zeros((1,nC),dtype=int), cs[:-consec_frames, :]])
        hit=(wsum >= consec_frames).any(axis=0).astype(int)

        if tid in set(early_tones):
            early=np.maximum(early, hit)
        if tid in set(late_tones):
            late=np.maximum(late, hit)

    out=np.array(["Neither"]*nC, dtype=object)
    out[(early==1) & (late==0)]="EarlyOnly"
    out[(early==0) & (late==1)]="LateOnly"
    out[(early==1) & (late==1)]="Overlap"
    return out

# -----------------------
# Freezing / tone metrics (tone-only freezing onsets)
# -----------------------
def tone_only_freeze_onsets(tone_flag: np.ndarray, freeze_flag: np.ndarray) -> np.ndarray:
    onsets=rising_edges(freeze_flag)
    return np.array([i for i in onsets if int(tone_flag[i])==1], dtype=int)

def compute_event_delta_mean(z: np.ndarray, onsets: np.ndarray, pre_frames: int, post_frames: int) -> np.ndarray:
    """Returns per-cell mean delta = mean(post) - mean(pre), averaged across events."""
    nT, nC = z.shape
    if len(onsets)==0:
        return np.full(nC, np.nan)
    deltas=[]
    for idx in onsets:
        if idx-pre_frames < 0 or idx+post_frames > nT:
            continue
        base=np.nanmean(z[idx-pre_frames:idx, :], axis=0)
        post=np.nanmean(z[idx:idx+post_frames, :], axis=0)
        deltas.append(post-base)
    if len(deltas)==0:
        return np.full(nC, np.nan)
    return np.nanmean(np.vstack(deltas), axis=0)

def compute_event_peak_mean(z: np.ndarray, onsets: np.ndarray, win_frames: int) -> np.ndarray:
    """Per event: peak (max) within [0..win_frames], then average peaks across events for each cell."""
    nT, nC = z.shape
    if len(onsets)==0:
        return np.full(nC, np.nan)
    peaks=[]
    for idx in onsets:
        if idx < 0 or idx+win_frames > nT:
            continue
        seg=z[idx:idx+win_frames, :]
        peaks.append(np.nanmax(seg, axis=0))
    if len(peaks)==0:
        return np.full(nC, np.nan)
    return np.nanmean(np.vstack(peaks), axis=0)

def event_triggered_traces(z: np.ndarray, onsets: np.ndarray, pre_frames: int, post_frames: int,
                           baseline_pre_frames: int | None=None) -> np.ndarray:
    """
    Returns array: n_events x n_cells x n_time
    If baseline_pre_frames is set, subtract per-event baseline mean over [-baseline_pre_frames..0].
    """
    nT, nC = z.shape
    kept=[]
    for idx in onsets:
        if idx-pre_frames < 0 or idx+post_frames >= nT:
            continue
        if baseline_pre_frames is not None and idx-baseline_pre_frames < 0:
            continue
        kept.append(idx)
    if len(kept)==0:
        return np.zeros((0, nC, pre_frames+post_frames+1), dtype=float)
    out=np.zeros((len(kept), nC, pre_frames+post_frames+1), dtype=float)
    for i, idx in enumerate(kept):
        seg=z[idx-pre_frames:idx+post_frames+1, :].T  # C x time
        if baseline_pre_frames is not None:
            base=np.nanmean(z[idx-baseline_pre_frames:idx, :], axis=0)  # C
            seg=seg - base[:,None]
        out[i,:,:]=seg
    return out
