import argparse, os
from utils_freezing import load_config
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    return ap.parse_args()

import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from utils_freezing import (
    extract_npz_parts, extract_npz_from_zip, extract_csvs_from_zip, choose_one_ext2_csv_per_animal,
    load_ext1_npz, load_ret_npz, load_ext2_session_csv,
    load_ext1_class_maps, load_ret_class_maps, classify_onset_evoked,
    infer_fps, rising_edges, ensure_dir, standardize_class_label
)

CLASSES=["EarlyOnly","Overlap","LateOnly"]
COMMON_T=np.arange(-5.0, 5.0+1e-9, 0.1)

def freeze_onsets_in_tone_subset(d, tone_subset: set) -> np.ndarray:
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

def per_animal_class_mean_traces(d, tone_cls_map_or_labels, tone_subset: set,
                                win_pre_s=5.0, win_post_s=5.0, baseline_pre_s=1.0):
    # tone_cls_map_or_labels: dict cell_id->class OR np.array aligned
    cell_ids=[str(x).strip() for x in d["cell_ids"]]
    if isinstance(tone_cls_map_or_labels, dict):
        tone_cls=np.array([standardize_class_label(tone_cls_map_or_labels.get(cid,"Neither")) for cid in cell_ids], dtype=object)
    else:
        tone_cls=np.array([standardize_class_label(x) for x in tone_cls_map_or_labels], dtype=object)

    fps=infer_fps(d["time"])
    if not np.isfinite(fps): fps=10.0
    preF=int(round(win_pre_s*fps)); postF=int(round(win_post_s*fps)); baseF=int(round(baseline_pre_s*fps))
    t_axis=np.arange(-preF, postF+1)/fps

    on=freeze_onsets_in_tone_subset(d, tone_subset)
    # keep only valid windows
    kept=[]
    for idx in on:
        if idx-preF<0 or idx+postF>=d["z"].shape[0] or idx-baseF<0:
            continue
        kept.append(idx)
    kept=np.array(kept, dtype=int)

    out={c: None for c in CLASSES}
    if len(kept)==0:
        return t_axis, out, {"n_events":0, "fps":fps, "n_cells":{c:int(np.sum(tone_cls==c)) for c in CLASSES}}

    # per class: per cell average across events, then mean across cells
    for c in CLASSES:
        idxs=np.where(tone_cls==c)[0]
        if len(idxs)==0:
            continue
        cell_tr=[]
        for cell in idxs:
            ev=[]
            for idx in kept:
                seg=d["z"][idx-preF:idx+postF+1, cell].astype(float)
                base=np.nanmean(d["z"][idx-baseF:idx, cell].astype(float))
                ev.append(seg-base)
            ev=np.vstack(ev)
            cell_tr.append(np.nanmean(ev, axis=0))
        cell_tr=np.vstack(cell_tr)
        out[c]=np.nanmean(cell_tr, axis=0)
    return t_axis, out, {"n_events":int(len(kept)), "fps":fps, "n_cells":{c:int(np.sum(tone_cls==c)) for c in CLASSES}}

def aggregate_animal_weighted(per_animal, cls):
    mats=[]
    for aid, t_axis, class_means in per_animal:
        tr=class_means.get(cls, None)
        if tr is None:
            continue
        mats.append(np.interp(COMMON_T, t_axis, tr))
    if len(mats)==0:
        return None
    A=np.vstack(mats)
    mean=np.nanmean(A, axis=0)
    sem=np.nanstd(A, axis=0, ddof=1)/np.sqrt(A.shape[0]) if A.shape[0]>1 else np.full_like(mean, np.nan)
    return mean, sem, A.shape[0]

def main(cfg):
    outdir=cfg["paths"]["output_dir"]
    ensure_dir(outdir)

    subset_defs=cfg["tone_subset_defs"]
    base_s=cfg["freezing_metrics"]["baseline_pre_s"]
    pre_s=cfg["freezing_metrics"]["trace_window_pre_s"]
    post_s=cfg["freezing_metrics"]["trace_window_post_s"]

    ext1_maps=load_ext1_class_maps(cfg["paths"]["ext1_classes_zip"], os.path.join(outdir,"_ext1_classes"))
    ret_maps=load_ret_class_maps(cfg["paths"]["ret_classes_zip"], os.path.join(outdir,"_ret_classes"))

    ext1_npz=extract_npz_parts(cfg["paths"]["ext1_zscored_npz_parts"], outdir)
    ret_npz=extract_npz_from_zip(cfg["paths"]["ret_zscored_npz_zip"], outdir)
    ext2_csvs=extract_csvs_from_zip(cfg["paths"]["ext2_raw_zip"], os.path.join(outdir,"_ext2_raw"))
    ext2_files=choose_one_ext2_csv_per_animal(ext2_csvs)

    p=cfg["tone_onset_classification_ext2"]

    session_anim={}
    # Ext1
    session_anim["Ext1"]=[]
    for npzpath in ext1_npz:
        d=load_ext1_npz(npzpath)
        aid=os.path.basename(npzpath).replace("_ext1_zscored_traces.npz","")
        cmap=ext1_maps.get(aid,{})
        session_anim["Ext1"].append((aid, d, cmap))
    # Ext2
    session_anim["Ext2"]=[]
    for fp in ext2_files:
        d=load_ext2_session_csv(fp)
        if d is None:
            continue
        aid=os.path.basename(fp).split("_")[0]
        cls=classify_onset_evoked(d["z"], d["tone_flag"], d["tone_id"], d["time"],
                                  thr_z=p["thr_z"], consec_frames=p["consec_frames"], window_s=p["window_s"],
                                  early_tones=p["early_tones"], late_tones=p["late_tones"])
        session_anim["Ext2"].append((aid, d, cls))
    # Retrieval
    session_anim["Retrieval"]=[]
    for npzpath in ret_npz:
        d=load_ret_npz(npzpath)
        aid=os.path.basename(npzpath).replace("_retrieval_zscored_traces.npz","")
        cmap=ret_maps.get(aid,{})
        session_anim["Retrieval"].append((aid, d, cmap))

    qc=[]
    for sess in ["Ext1","Ext2","Retrieval"]:
        for subset_name, tones in subset_defs[sess].items():
            per_animal=[]
            for aid, d, clsinfo in session_anim[sess]:
                t_axis, class_means, info=per_animal_class_mean_traces(d, clsinfo, set(tones),
                                                                       win_pre_s=pre_s, win_post_s=post_s, baseline_pre_s=base_s)
                per_animal.append((aid, t_axis, class_means))
                qc.append({"session":sess,"subset":subset_name,"animal_id":aid,"n_freeze_onsets_tone_subset":info["n_events"],
                           "n_cells_EarlyOnly":info["n_cells"]["EarlyOnly"],"n_cells_Overlap":info["n_cells"]["Overlap"],"n_cells_LateOnly":info["n_cells"]["LateOnly"],"fps":info["fps"]})

            # plot overlay
            fig, ax=plt.subplots(figsize=(8.6,5.1))
            for c in CLASSES:
                agg=aggregate_animal_weighted(per_animal, c)
                if agg is None:
                    continue
                mean, sem, nA = agg
                ax.plot(COMMON_T, mean, linewidth=2, label=f"{c} (n={nA})")
                ax.fill_between(COMMON_T, mean-sem, mean+sem, alpha=0.18)
                pd.DataFrame({"time_s":COMMON_T, "mean":mean, "sem":sem}).to_csv(
                    os.path.join(outdir, f"{sess}_{subset_name}_freezeOnsetDeltaZ_trace_{c}_ANIMALWEIGHTED.csv"), index=False
                )
            ax.axvline(0, alpha=0.6); ax.axhline(0, alpha=0.25)
            ax.set_xlabel("Time from freezing onset (s)")
            ax.set_ylabel("Δz (baseline: -1..0 s)")
            ax.set_title(f"{sess}: freezing-onset Δz mean trace (tone-only)\n{subset_name} | split by tone-onset class (animal-weighted)")
            ax.legend(frameon=False)
            ax.set_ylim(None, None)
            ax.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{sess}_{subset_name}_freezeOnsetDeltaZ_trace_splitByClass_ANIMALWEIGHTED.png"), dpi=300)
            plt.close()

    pd.DataFrame(qc).to_csv(os.path.join(outdir,"freezeOnsetDeltaZ_traceToneSubsets_splitByClass_QCcounts.csv"), index=False)

if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args.config)
    main(cfg)
