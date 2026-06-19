#!/usr/bin/env python3
"""
00_retrieval_process_raw_traces.py

Inputs:
  - Retrieval raw trace zip (CSV files per animal)

Outputs:
  - ret_classes_proportions_only.zip
      * Ret_classes_proportions_perAnimal.csv
      * Ret_cellClassifications_long.csv
      * {animal}_ret_onset_evoked_cell_classes.csv (per animal)

Classification rule (same as Ext2):
  - z-score each cell across entire session
  - determine tone onset frame for Tone 1 and Tone 4
  - delta-from-onset: z(t) - z(onset)
  - onset-evoked if delta >= 0.5 z for >= 10 consecutive frames within 0–3s post onset
  - group labels:
      EarlyOnly: Tone1 evoked only
      LateOnly:  Tone4 evoked only
      Overlap:   both
      Neither:   neither

Notes:
  - Best-effort column detection for tone/freezing; if fps can't be inferred, assumes 10 Hz.
"""
import os, re, glob, zipfile, json, argparse, shutil
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES = ["EarlyOnly","Overlap","LateOnly","Neither"]

def score_ret_raw_zip(zip_path: str) -> tuple[int, int]:
    animal_ids=set()
    csv_count=0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            base=os.path.basename(name)
            if not base.lower().endswith(".csv"):
                continue
            if "__macosx" in name.lower() or base.startswith("._"):
                continue
            csv_count += 1
            m=re.search(r"(?:animal)?(\d{1,5})", base, flags=re.IGNORECASE)
            if m:
                animal_ids.add(m.group(1))
    return len(animal_ids), csv_count

def resolve_ret_raw_zip(data_dir: str, configured_name: str) -> str:
    seen=set()
    candidates=[]
    for name in [configured_name, "ret_raw_trace_files 2.zip", "ret_raw_trace_files.zip"]:
        path=os.path.join(data_dir, name)
        if os.path.exists(path) and path not in seen:
            candidates.append(path)
            seen.add(path)
    for path in sorted(glob.glob(os.path.join(data_dir, "ret_raw_trace_files*.zip"))):
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    if not candidates:
        raise FileNotFoundError("No retrieval raw zip found")
    configured_path=os.path.join(data_dir, configured_name)
    ranked=[]
    for path in candidates:
        ranked.append((score_ret_raw_zip(path), path == configured_path, path))
    ranked.sort(key=lambda item: (item[0][0], item[0][1], item[1]), reverse=True)
    return ranked[0][2]

def ensure_ret_raw_extracted(ret_zip: str, tmp_dir: str) -> None:
    marker=os.path.join(tmp_dir, ".source_zip")
    current=os.path.abspath(ret_zip)
    existing_csvs=glob.glob(os.path.join(tmp_dir, "**", "*.csv"), recursive=True)
    previous=None
    if os.path.exists(marker):
        with open(marker, "r", encoding="utf-8") as fh:
            previous=fh.read().strip()
    if existing_csvs and previous == current:
        return
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ret_zip, "r") as zf:
        zf.extractall(tmp_dir)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(current)

def load_config(cfg_path):
    with open(cfg_path,"r") as f:
        return json.load(f)

def sex_of(animal_id, female_set, male_set):
    a=str(animal_id)
    if a in female_set: return "Female"
    if a in male_set: return "Male"
    return "Unknown"

def infer_animal_id_from_path(path:str)->str:
    base=os.path.basename(path)
    m=re.search(r"(?:animal)?(\d{1,5})", base, flags=re.IGNORECASE)
    if m: return m.group(1)
    return base.split("_")[0].split(".")[0]

def normalize_cell_id_any(cid: str, width: int = 3) -> str:
    s=str(cid).strip()
    ms=re.findall(r"(\d+)", s)
    if ms:
        return "C"+ms[-1].zfill(width)
    return s

def detect_columns(df: pd.DataFrame):
    cols=[str(c).strip() for c in df.columns]
    df.columns=cols
    lc={c:c.lower() for c in cols}
    time_col=None
    for c in cols:
        if "time" in lc[c]:
            time_col=c
            break
    tone_id=None
    for c in cols:
        if lc[c] in ("cs","tone_id","toneid","cs_id","toneindex","tone_index"):
            tone_id=c; break
    tone_flag=None
    for c in cols:
        l=lc[c]
        if ("tone" in l or "cs" in l) and any(k in l for k in ("flag","within","is_","in_","intone","in_tone","toneon","tone_on")):
            tone_flag=c; break
    freeze=None
    for c in cols:
        if "freez" in lc[c]:
            freeze=c; break

    meta=set([x for x in [time_col,tone_id,tone_flag,freeze] if x is not None])
    numeric_cols=[c for c in cols if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    # Prefer C### columns by name even if dtype is object (we'll coerce later)
    name_cells=[c for c in cols if c not in meta and re.match(r"^C\\d+", c, flags=re.IGNORECASE)]
    # Handle numeric-only or 'undecided' columns (e.g., 10489 retrieval)
    extra_name_cells=[c for c in cols if c not in meta and (re.match(r"^\\d+$", c.strip()) or lc[c].startswith("undecided"))]
    cell_cols=name_cells + [c for c in extra_name_cells if c not in name_cells]
    if len(cell_cols) < max(5, int(0.4*len(numeric_cols))):
        cell_cols=numeric_cols
    return time_col, tone_id, tone_flag, freeze, cell_cols

def get_fps(time_vec):
    dt=np.diff(time_vec.astype(float))
    dt=dt[np.isfinite(dt) & (dt>0)]
    if len(dt)==0: return None
    med=float(np.median(dt))
    return 1.0/med if med>0 else None

def find_tone_epochs(df, tone_id_col, tone_flag_col):
    n=len(df)
    if tone_id_col is not None and tone_id_col in df.columns:
        tid=pd.to_numeric(df[tone_id_col].values, errors="coerce")
        uniq=[int(x) for x in np.unique(tid[np.isfinite(tid)])]
        uniq=[u for u in uniq if u!=0]
        if len(uniq)>0:
            epochs=[]
            for u in sorted(uniq):
                idxs=np.where(tid==u)[0]
                if len(idxs)==0: continue
                epochs.append({"tone_num":u,"onset_idx":int(idxs[0]),"idxs":idxs})
            epochs=sorted(epochs, key=lambda d:d["onset_idx"])
            for k,e in enumerate(epochs, start=1): e["tone_order"]=k
            return epochs
    if tone_flag_col is None or tone_flag_col not in df.columns:
        raise ValueError("No tone_id or tone_flag columns")
    flag=pd.to_numeric(df[tone_flag_col].values, errors="coerce")
    flag=np.where(np.isfinite(flag), flag, 0.0)
    flag=(flag>0).astype(int)
    rises=np.where((flag[1:]==1)&(flag[:-1]==0))[0]+1
    falls=np.where((flag[1:]==0)&(flag[:-1]==1))[0]+1
    if flag[0]==1: rises=np.r_[0,rises]
    if flag[-1]==1: falls=np.r_[falls,n]
    epochs=[]
    for k,(s,e) in enumerate(zip(rises,falls), start=1):
        idxs=np.arange(s,e)
        epochs.append({"tone_num":k,"tone_order":k,"onset_idx":int(s),"idxs":idxs})
    return epochs

def zscore_matrix(X):
    mu=np.nanmean(X, axis=0)
    sd=np.nanstd(X, axis=0)
    sd=np.where(sd==0, np.nan, sd)
    return ((X-mu)/sd).astype(np.float32)

def is_onset_evoked(z_trace, onset_idx, fps, zthr=0.5, consec=10, post_s=3.0):
    if fps is None or fps<=0:
        fps=10.0
    post_frames=int(round(post_s*fps))
    start=onset_idx+1
    end=min(len(z_trace), onset_idx+post_frames+1)
    if end-start < consec:
        return False
    delta=z_trace[start:end] - float(z_trace[onset_idx])
    ok=(delta>=zthr).astype(int)
    run=0
    for v in ok:
        if v:
            run += 1
            if run>=consec:
                return True
        else:
            run=0
    return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pipeline_config.json")
    args=ap.parse_args()

    cfg=load_config(args.config)
    data_dir=cfg["paths"]["data_dir"]
    ret_zip=resolve_ret_raw_zip(data_dir, cfg["paths"]["ret_raw_zip"])

    female_set=set(cfg["cohort_sex_map"]["female"])
    male_set=set(cfg["cohort_sex_map"]["male"])

    rule=cfg["classification"]["onset_evoked_rule"]
    zthr=float(rule["z_threshold"])
    consec=int(rule["consecutive_frames"])
    post_s=float(rule["post_onset_seconds"])
    early_tone=int(cfg["classification"]["retrieval_early_tone"])
    late_tone=int(cfg["classification"]["retrieval_late_tone"])

    tmp_dir=os.path.join(data_dir,"_tmp_ret_raw")
    ensure_ret_raw_extracted(ret_zip, tmp_dir)

    candidates=sorted(glob.glob(os.path.join(tmp_dir,"**","*.csv"), recursive=True))
    classes_dir=os.path.join(data_dir, cfg["outputs"]["ret_classes_dir"])
    Path(classes_dir).mkdir(parents=True, exist_ok=True)

    per_rows=[]
    long_rows=[]

    for fp in candidates:
        try:
            df=pd.read_csv(fp)
        except Exception:
            continue
        if df.shape[0] < 20 or df.shape[1] < 5:
            continue

        animal=infer_animal_id_from_path(fp)
        sex=sex_of(animal, female_set, male_set)

        time_col,tone_id_col,tone_flag_col,freeze_col,cell_cols=detect_columns(df)
        if len(cell_cols) < 5:
            continue

        fps=None
        if time_col is not None:
            t=pd.to_numeric(df[time_col].values, errors="coerce")
            if not np.all(np.isnan(t)):
                fps=get_fps(t)

        try:
            epochs=find_tone_epochs(df, tone_id_col, tone_flag_col)
        except Exception:
            continue
        epochs=sorted(epochs, key=lambda d:d["onset_idx"])
        if len(epochs) < max(early_tone, late_tone):
            continue

        onset_early=int(epochs[early_tone-1]["onset_idx"])
        onset_late=int(epochs[late_tone-1]["onset_idx"])

        # Coerce all cell columns to numeric to avoid dropping cells with object dtype
        df[cell_cols]=df[cell_cols].apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",","", regex=False).str.strip(), errors="coerce"))
        X=df[cell_cols].to_numpy(dtype=float)
        # Drop all-NaN columns (empty/invalid cells)
        valid_mask=~np.all(np.isnan(X), axis=0)
        X=X[:, valid_mask]
        cell_ids=[]
        for idx, c in enumerate(cell_cols):
            norm=normalize_cell_id_any(c,3)
            if not re.match(r"^C\\d+", norm, flags=re.IGNORECASE):
                norm=f"C{idx:03d}"
            cell_ids.append(norm)
        cell_ids=[cid for cid, keep in zip(cell_ids, valid_mask) if keep]
        if X.shape[1] == 0:
            continue
        Z=zscore_matrix(X)

        resp_early=np.array([is_onset_evoked(Z[:,j], onset_early, fps, zthr, consec, post_s) for j in range(Z.shape[1])], dtype=bool)
        resp_late =np.array([is_onset_evoked(Z[:,j], onset_late,  fps, zthr, consec, post_s) for j in range(Z.shape[1])], dtype=bool)

        classes=np.where(resp_early & ~resp_late, "EarlyOnly",
                 np.where(~resp_early & resp_late, "LateOnly",
                 np.where(resp_early & resp_late, "Overlap", "Neither")))

        cls_df=pd.DataFrame({"cell_id":cell_ids,"class":classes})
        cls_path=os.path.join(classes_dir, f"{animal}_ret_onset_evoked_cell_classes.csv")
        cls_df.to_csv(cls_path, index=False)

        counts=cls_df["class"].value_counts().reindex(CLASSES, fill_value=0)
        N=int(counts.sum())
        per_rows.append({
            "animal_id":animal,"sex":sex,"n_cells":N,
            **{f"n_{k}":int(v) for k,v in counts.items()},
            **{f"p_{k}":float(v/N) if N>0 else np.nan for k,v in counts.items()}
        })

        tmp=cls_df.copy()
        tmp["animal_id"]=animal
        tmp["sex"]=sex
        long_rows.append(tmp)

    per_anim=pd.DataFrame(per_rows).sort_values("animal_id")
    long=pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

    per_path=os.path.join(data_dir, cfg["outputs"]["ret_summary_csv"])
    long_path=os.path.join(data_dir, cfg["outputs"]["ret_cell_long_csv"])
    per_anim.to_csv(per_path, index=False)
    long.to_csv(long_path, index=False)

    out_zip=os.path.join(data_dir, "ret_classes_proportions_only.zip")
    with zipfile.ZipFile(out_zip,"w",zipfile.ZIP_DEFLATED) as zf:
        zf.write(per_path, arcname=os.path.basename(per_path))
        zf.write(long_path, arcname=os.path.basename(long_path))
        for fp in sorted(glob.glob(os.path.join(classes_dir,"*_ret_onset_evoked_cell_classes.csv"))):
            zf.write(fp, arcname=os.path.basename(fp))

    print("Wrote:", out_zip)

if __name__=="__main__":
    main()
