#!/usr/bin/env python3
"""
01_retrieval_tone_by_tone_activity_freeze.py

Uses:
  - per-animal per-cell classes from 00_* (ret_classes directory)
  - retrieval raw trace CSVs (same zip)

Outputs:
  - Ret_toneByTone_activity_freezing_perAnimal.csv
  - Ret_toneByTone_summary_meanSEM.csv
  - Ret_toneByTone_activityLines_freezeBars_{ALL,Male,Female}.png
  - ret_toneByTone_activity_freeze_bundle.zip

Definitions:
  - EarlyOnly/LateOnly from Tone1 vs Tone4 onset-evoked classification.
  - For each tone epoch, compute mean z activity during that epoch for EarlyOnly and LateOnly
    separately within each animal.
  - Freezing per tone is mean freezing flag during the tone epoch (fraction).
  - Aggregate is animal-weighted (each animal contributes one mean per tone).
"""
import os, re, glob, zipfile, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_config(cfg_path):
    with open(cfg_path,"r") as f:
        return json.load(f)

def sex_of(animal_id, female_set, male_set):
    a=str(animal_id)
    if a in female_set: return "Female"
    if a in male_set: return "Male"
    return "Unknown"

def normalize_cell_id_any(cid: str, width: int = 3) -> str:
    s=str(cid).strip()
    ms=re.findall(r"(\d+)", s)
    if ms:
        return "C"+ms[-1].zfill(width)
    return s

def zscore_matrix(X):
    mu=np.nanmean(X,axis=0)
    sd=np.nanstd(X,axis=0)
    sd=np.where(sd==0, np.nan, sd)
    return ((X-mu)/sd).astype(np.float32)

def detect_cols(df):
    cols=[str(c).strip() for c in df.columns]
    df.columns=cols
    tone_id=None
    for c in cols:
        if c.lower() in ("cs","tone_id","toneid","toneindex","tone_index"):
            tone_id=c; break
    tone_flag=None
    for c in cols:
        lc=c.lower()
        if ("tone" in lc or "cs" in lc) and any(k in lc for k in ("flag","within","is_","in_")):
            tone_flag=c; break
    freeze=None
    for c in cols:
        if "freez" in c.lower():
            freeze=c; break
    cell_cols=[c for c in cols if re.match(r"^C\d+", c, flags=re.IGNORECASE)]
    if len(cell_cols)<5:
        numeric=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        numeric=[c for c in numeric if c not in set([tone_id,tone_flag,freeze])]
        cell_cols=numeric
    return tone_id, tone_flag, freeze, cell_cols

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
        raise ValueError("No tone_id or tone_flag")
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

def mean_sem(x):
    x=np.asarray(x, float)
    x=x[np.isfinite(x)]
    if len(x)==0: return np.nan, np.nan, 0
    sem=float(np.std(x, ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.0
    return float(np.mean(x)), sem, int(len(x))

def plot_tone_by_tone(summary_df, sex_label, out_png):
    s=summary_df[summary_df["sex"]==sex_label].sort_values("tone")
    x=s["tone"].values
    fig, ax1=plt.subplots(figsize=(9,4.8))
    ax1.bar(x, s["freeze_mean"].values, yerr=s["freeze_sem"].values, capsize=4)
    ax1.set_xlabel("Tone #")
    ax1.set_ylabel("Freezing (fraction during tone)")
    ax1.set_xticks(x); ax1.set_ylim(0,1)
    ax2=ax1.twinx()
    ax2.errorbar(x, s["early_z_mean"].values, yerr=s["early_z_sem"].values, marker="o", linewidth=2, label="EarlyOnly")
    ax2.errorbar(x, s["late_z_mean"].values, yerr=s["late_z_sem"].values, marker="o", linewidth=2, label="LateOnly")
    ax2.set_ylabel("Mean z activity during tone")
    ax2.axhline(0, linewidth=1, alpha=0.4)
    ax2.legend(frameon=False, loc="upper right")
    ax1.set_title(f"Retrieval tone-by-tone activity with freezing ({sex_label})\nAnimal-weighted mean ± SEM")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pipeline_config.json")
    args=ap.parse_args()
    cfg=load_config(args.config)
    data_dir=cfg["paths"]["data_dir"]

    female_set=set(cfg["cohort_sex_map"]["female"])
    male_set=set(cfg["cohort_sex_map"]["male"])

    ret_zip=os.path.join(data_dir, cfg["paths"]["ret_raw_zip"])
    tmp_dir=os.path.join(data_dir,"_tmp_ret_raw")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    if not glob.glob(os.path.join(tmp_dir,"**","*.csv"), recursive=True):
        with zipfile.ZipFile(ret_zip,"r") as zf:
            zf.extractall(tmp_dir)

    class_dir=os.path.join(data_dir, cfg["outputs"]["ret_classes_dir"])
    class_files=sorted(glob.glob(os.path.join(class_dir,"*_ret_onset_evoked_cell_classes.csv")))
    class_map={}
    for fp in class_files:
        m=re.search(r"(\d{1,5})", os.path.basename(fp))
        if not m: 
            continue
        animal=m.group(1)
        d=pd.read_csv(fp)
        d["cell_id"]=d["cell_id"].astype(str)
        class_map[animal]=d.set_index("cell_id")["class"].to_dict()

    rows=[]
    for animal, cmap in sorted(class_map.items(), key=lambda kv:int(kv[0])):
        hits=glob.glob(os.path.join(tmp_dir,"**",f"*{animal}*csv"), recursive=True)
        hits=[h for h in hits if "ret_aligned_toneflag" not in os.path.basename(h).lower()]
        if not hits:
            continue
        fp=sorted(hits, key=lambda p:(len(os.path.basename(p)), os.path.basename(p)))[0]
        df=pd.read_csv(fp)
        df.columns=[str(c).strip() for c in df.columns]

        tone_id_col, tone_flag_col, freeze_col, cell_cols = detect_cols(df)
        if len(cell_cols)<5:
            continue

        epochs=find_tone_epochs(df, tone_id_col, tone_flag_col)
        epochs=sorted(epochs, key=lambda d:d["onset_idx"])[:4]
        if len(epochs)<4:
            continue

        freeze = pd.to_numeric(df[freeze_col].values, errors="coerce") if freeze_col is not None else np.zeros(len(df))
        freeze = np.where(np.isfinite(freeze), freeze, 0.0)
        freeze = (freeze>0).astype(int)

        X=df[cell_cols].to_numpy(dtype=float)
        Z=zscore_matrix(X)
        cell_ids=[normalize_cell_id_any(c,3) for c in cell_cols]
        cls_arr=np.array([cmap.get(cid, "NA") for cid in cell_ids], dtype=object)
        early_mask=cls_arr=="EarlyOnly"
        late_mask=cls_arr=="LateOnly"

        early_means=[]; late_means=[]; freeze_means=[]
        for e in epochs:
            idxs=e["idxs"]
            freeze_means.append(float(np.mean(freeze[idxs])) if len(idxs)>0 else np.nan)
            early_means.append(float(np.nanmean(Z[idxs][:,early_mask])) if early_mask.any() else np.nan)
            late_means.append(float(np.nanmean(Z[idxs][:,late_mask])) if late_mask.any() else np.nan)

        rows.append({
            "animal_id":animal,
            "sex":sex_of(animal, female_set, male_set),
            "raw_file":os.path.basename(fp),
            "n_early_cells":int(early_mask.sum()),
            "n_late_cells":int(late_mask.sum()),
            **{f"early_z_tone{t}":early_means[t-1] for t in range(1,5)},
            **{f"late_z_tone{t}":late_means[t-1] for t in range(1,5)},
            **{f"freeze_tone{t}":freeze_means[t-1] for t in range(1,5)},
        })

    tone_df=pd.DataFrame(rows).sort_values("animal_id")
    tone_csv=os.path.join(data_dir,"Ret_toneByTone_activity_freezing_perAnimal.csv")
    tone_df.to_csv(tone_csv, index=False)

    summary_rows=[]
    for sex in ["All","Male","Female"]:
        sub=tone_df if sex=="All" else tone_df[tone_df["sex"]==sex]
        for t in range(1,5):
            m_f, sem_f, n_f = mean_sem(sub[f"freeze_tone{t}"])
            m_e, sem_e, n_e = mean_sem(sub[f"early_z_tone{t}"])
            m_l, sem_l, n_l = mean_sem(sub[f"late_z_tone{t}"])
            summary_rows.append({
                "sex":sex,"tone":t,
                "freeze_mean":m_f,"freeze_sem":sem_f,"freeze_n":n_f,
                "early_z_mean":m_e,"early_z_sem":sem_e,"early_z_n":n_e,
                "late_z_mean":m_l,"late_z_sem":sem_l,"late_z_n":n_l,
            })
    summary=pd.DataFrame(summary_rows)
    summary_csv=os.path.join(data_dir,"Ret_toneByTone_summary_meanSEM.csv")
    summary.to_csv(summary_csv, index=False)

    png_all=os.path.join(data_dir,"Ret_toneByTone_activityLines_freezeBars_ALL.png")
    png_m  =os.path.join(data_dir,"Ret_toneByTone_activityLines_freezeBars_Male.png")
    png_f  =os.path.join(data_dir,"Ret_toneByTone_activityLines_freezeBars_Female.png")
    plot_tone_by_tone(summary,"All",png_all)
    plot_tone_by_tone(summary,"Male",png_m)
    plot_tone_by_tone(summary,"Female",png_f)

    out_zip=os.path.join(data_dir,"ret_toneByTone_activity_freeze_bundle.zip")
    with zipfile.ZipFile(out_zip,"w",zipfile.ZIP_DEFLATED) as zf:
        for fp in [tone_csv, summary_csv, png_all, png_m, png_f]:
            zf.write(fp, arcname=os.path.basename(fp))

    print("Wrote:", out_zip)

if __name__=="__main__":
    main()
