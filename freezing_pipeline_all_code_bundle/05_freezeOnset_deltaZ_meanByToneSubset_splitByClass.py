import argparse, os
from utils_freezing import load_config
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    return ap.parse_args()

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from utils_freezing import (
    extract_npz_parts, extract_npz_from_zip, extract_csvs_from_zip, choose_one_ext2_csv_per_animal,
    load_ext1_npz, load_ret_npz, load_ext2_session_csv,
    load_ext1_class_maps, load_ret_class_maps, classify_onset_evoked,
    infer_fps, rising_edges, ensure_dir, standardize_class_label
)

CLASSES=["EarlyOnly","Overlap","LateOnly"]

def freeze_onsets_in_tone_subset(d, tone_subset: set) -> np.ndarray:
    # rising edges in FreezeFlag, restricted to ToneFlag==1 and ToneIndex in subset
    on=np.where((d["freeze"][1:]==1) & (d["freeze"][:-1]==0))[0]+1
    if d["freeze"][0]==1:
        on=np.r_[0,on]
    keep=[]
    for idx in on:
        if int(d["tone_flag"][idx])!=1:
            continue
        tid=int(d["tone_id"][idx]) if "tone_id" in d else 0
        if tid in tone_subset:
            keep.append(idx)
    return np.array(keep, dtype=int)

def mean_delta_per_cell(d, onsets, baseline_pre_s, post_s):
    fps=infer_fps(d["time"])
    if not np.isfinite(fps): fps=10.0
    pre=int(round(baseline_pre_s*fps))
    post=int(round(post_s*fps))
    z=d["z"]
    nT,nC=z.shape
    deltas=[]
    for idx in onsets:
        if idx-pre<0 or idx+post>nT:
            continue
        base=np.nanmean(z[idx-pre:idx,:], axis=0)
        postm=np.nanmean(z[idx:idx+post,:], axis=0)
        deltas.append(postm-base)
    if len(deltas)==0:
        return np.full(nC, np.nan), 0
    D=np.vstack(deltas)
    return np.nanmean(D, axis=0), D.shape[0]

def main(cfg):
    outdir=cfg["paths"]["output_dir"]
    ensure_dir(outdir)

    ext1_maps=load_ext1_class_maps(cfg["paths"]["ext1_classes_zip"], os.path.join(outdir,"_ext1_classes"))
    ret_maps=load_ret_class_maps(cfg["paths"]["ret_classes_zip"], os.path.join(outdir,"_ret_classes"))
    ext1_npz=extract_npz_parts(cfg["paths"]["ext1_zscored_npz_parts"], outdir)
    ret_npz=extract_npz_from_zip(cfg["paths"]["ret_zscored_npz_zip"], outdir)
    ext2_csvs=extract_csvs_from_zip(cfg["paths"]["ext2_raw_zip"], os.path.join(outdir,"_ext2_raw"))
    ext2_files=choose_one_ext2_csv_per_animal(ext2_csvs)

    p=cfg["tone_onset_classification_ext2"]
    base_s=cfg["freezing_metrics"]["baseline_pre_s"]
    post_s=cfg["freezing_metrics"]["post_s"]
    subset_defs=cfg["tone_subset_defs"]

    # Build session animals list
    sessions=[]

    # Ext1
    ext1=[]
    for npzpath in ext1_npz:
        d=load_ext1_npz(npzpath)
        aid=os.path.basename(npzpath).replace("_ext1_zscored_traces.npz","")
        cmap=ext1_maps.get(aid,{})
        tone_cls=np.array([standardize_class_label(cmap.get(cid,"Neither")) for cid in d["cell_ids"]], dtype=object)
        ext1.append((aid,d,tone_cls))
    sessions.append(("Ext1", ext1))

    # Ext2
    ext2=[]
    for fp in ext2_files:
        d=load_ext2_session_csv(fp)
        if d is None:
            continue
        aid=os.path.basename(fp).split("_")[0]
        tone_cls=classify_onset_evoked(d["z"], d["tone_flag"], d["tone_id"], d["time"],
                                       thr_z=p["thr_z"], consec_frames=p["consec_frames"], window_s=p["window_s"],
                                       early_tones=p["early_tones"], late_tones=p["late_tones"])
        ext2.append((aid,d,tone_cls))
    sessions.append(("Ext2", ext2))

    # Retrieval
    ret=[]
    for npzpath in ret_npz:
        d=load_ret_npz(npzpath)
        aid=os.path.basename(npzpath).replace("_retrieval_zscored_traces.npz","")
        cmap=ret_maps.get(aid,{})
        tone_cls=np.array([standardize_class_label(cmap.get(cid,"Neither")) for cid in d["cell_ids"]], dtype=object)
        ret.append((aid,d,tone_cls))
    sessions.append(("Retrieval", ret))

    rows=[]
    qc=[]
    for sess, animals in sessions:
        for subset_name, tones in subset_defs[sess].items():
            tone_set=set(tones)
            for aid, d, tone_cls in animals:
                on=freeze_onsets_in_tone_subset(d, tone_set)
                deltas, n_events=mean_delta_per_cell(d, on, base_s, post_s)
                qc.append({"session":sess,"subset":subset_name,"animal_id":aid,"n_freeze_onsets_tone_subset":n_events})
                for c in CLASSES:
                    mask=(tone_cls==c)
                    val=float(np.nanmean(deltas[mask])) if np.any(mask) else np.nan
                    rows.append({"session":sess,"subset":subset_name,"animal_id":aid,"class":c,"mean_freeze_deltaZ":val,"n_cells_in_class":int(np.sum(mask)),"n_freeze_onsets_tone_subset":n_events})

    out_df=pd.DataFrame(rows)
    qc_df=pd.DataFrame(qc)
    out_df.to_csv(os.path.join(outdir,"freezeOnset_deltaZ_meanByToneSubset_byClass_perAnimal.csv"), index=False)
    qc_df.to_csv(os.path.join(outdir,"freezeOnset_deltaZ_eventCounts_byToneSubset_perAnimal.csv"), index=False)

    # Plot 1 per session (animal-weighted bars) with early vs late subset, faceted by class
    for sess, _ in sessions:
        df=out_df[out_df["session"]==sess].copy()
        # per subset x class animal mean ± sem
        fig, ax=plt.subplots(figsize=(8.8,5.0))
        subsets=list(subset_defs[sess].keys())
        x=np.arange(len(subsets))
        width=0.22
        for j,c in enumerate(CLASSES):
            means=[]; sems=[]
            for sname in subsets:
                vals=df[(df["subset"]==sname)&(df["class"]==c)]["mean_freeze_deltaZ"].astype(float).values
                vals=vals[np.isfinite(vals)]
                means.append(np.nanmean(vals) if len(vals)>0 else np.nan)
                sems.append(np.nanstd(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else np.nan)
            ax.bar(x + (j-1)*width, means, width=width, alpha=0.85, label=c)
            ax.errorbar(x + (j-1)*width, means, yerr=sems, fmt="none", capsize=4, linewidth=1.3, alpha=0.9)
        ax.axhline(0, alpha=0.2)
        ax.set_xticks(x); ax.set_xticklabels(subsets)
        ax.set_ylabel("Mean freezing-onset Δz (post1s - pre1s)")
        ax.set_title(f"{sess}: mean freezing Δz by tone subset, split by tone-onset class")
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{sess}_freezeOnsetDeltaZ_mean_byToneSubset_splitByClass_ANIMALWEIGHTED.png"), dpi=300)
        plt.close()

if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args.config)
    main(cfg)
