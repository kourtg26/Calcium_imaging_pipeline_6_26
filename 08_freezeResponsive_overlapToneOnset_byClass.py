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
    infer_fps, tone_only_freeze_onsets, compute_event_delta_mean, ensure_dir, standardize_class_label
)

CLASSES=["EarlyOnly","Overlap","LateOnly"]

def main(cfg):
    outdir=cfg["paths"]["output_dir"]
    ensure_dir(outdir)

    ext1_maps=load_ext1_class_maps(cfg["paths"]["ext1_classes_zip"], os.path.join(outdir,"_ext1_classes"))
    ret_maps=load_ret_class_maps(cfg["paths"]["ret_classes_zip"], os.path.join(outdir,"_ret_classes"))

    ext1_npz=extract_npz_parts(cfg["paths"]["ext1_zscored_npz_parts"], outdir)
    ret_npz=extract_npz_from_zip(cfg["paths"]["ret_zscored_npz_zip"], outdir)
    ext2_csvs=extract_csvs_from_zip(cfg["paths"]["ext2_raw_zip"], os.path.join(outdir,"_ext2_raw"))
    ext2_files=choose_one_ext2_csv_per_animal(ext2_csvs)

    base_s=cfg["freezing_metrics"]["baseline_pre_s"]
    post_s=cfg["freezing_metrics"]["post_s"]
    p=cfg["tone_onset_classification_ext2"]

    thresholds=[0.0, -0.5]  # any decrease; strong decrease

    rows=[]
    qc=[]
    plot_files=[]
    rng=np.random.default_rng(0)

    def add_rows(session, aid, d, tone_cls, thr):
        fps=infer_fps(d["time"])
        if not np.isfinite(fps): fps=10.0
        preF=int(round(base_s*fps))
        postF=int(round(post_s*fps))
        freeze_on=tone_only_freeze_onsets(d["tone_flag"], d["freeze"])
        delta=compute_event_delta_mean(d["z"], freeze_on, preF, postF)
        qc.append({"session":session,"animal_id":aid,"n_toneFreezeOnsets_used":int(len(freeze_on)),"fps":fps})

        tone_cls=np.array([standardize_class_label(x) for x in tone_cls], dtype=object)
        freeze_resp=np.isfinite(delta) & (delta <= thr)
        N_freeze=int(np.sum(freeze_resp))
        N_by={c:int(np.sum(freeze_resp & (tone_cls==c))) for c in CLASSES}
        N_any=int(sum(N_by.values()))
        N_not=N_freeze - N_any

        row={
            "session":session, "animal_id":aid, "freeze_resp_threshold":thr,
            "n_toneFreezeOnsets_used": int(len(freeze_on)),
            "n_cells_total": int(len(delta)),
            "n_freeze_responsive": N_freeze,
            "n_freeze_resp_and_anyToneResp": N_any,
            "n_freeze_resp_and_EarlyOnly": N_by["EarlyOnly"],
            "n_freeze_resp_and_Overlap": N_by["Overlap"],
            "n_freeze_resp_and_LateOnly": N_by["LateOnly"],
            "n_freeze_resp_notToneResp": N_not
        }
        if N_freeze>0:
            row.update({
                "p_anyTone_within_freezeResp": N_any/N_freeze,
                "p_EarlyOnly_within_freezeResp": N_by["EarlyOnly"]/N_freeze,
                "p_Overlap_within_freezeResp": N_by["Overlap"]/N_freeze,
                "p_LateOnly_within_freezeResp": N_by["LateOnly"]/N_freeze,
                "p_notToneResp_within_freezeResp": N_not/N_freeze,
            })
            if N_any>0:
                row.update({
                    "p_EarlyOnly_within_freezeRespToneResp": N_by["EarlyOnly"]/N_any,
                    "p_Overlap_within_freezeRespToneResp": N_by["Overlap"]/N_any,
                    "p_LateOnly_within_freezeRespToneResp": N_by["LateOnly"]/N_any,
                })
            else:
                row.update({
                    "p_EarlyOnly_within_freezeRespToneResp": np.nan,
                    "p_Overlap_within_freezeRespToneResp": np.nan,
                    "p_LateOnly_within_freezeRespToneResp": np.nan,
                })
        else:
            row.update({k: np.nan for k in [
                "p_anyTone_within_freezeResp","p_EarlyOnly_within_freezeResp","p_Overlap_within_freezeResp",
                "p_LateOnly_within_freezeResp","p_notToneResp_within_freezeResp",
                "p_EarlyOnly_within_freezeRespToneResp","p_Overlap_within_freezeRespToneResp","p_LateOnly_within_freezeRespToneResp"
            ]})
        rows.append(row)

    # Ext1
    for npzpath in ext1_npz:
        d=load_ext1_npz(npzpath)
        aid=os.path.basename(npzpath).replace("_ext1_zscored_traces.npz","")
        cmap=ext1_maps.get(aid,{})
        tone_cls=[cmap.get(cid,"Neither") for cid in d["cell_ids"]]
        for thr in thresholds:
            add_rows("Ext1", aid, d, tone_cls, thr)

    # Ext2
    for fp in ext2_files:
        d=load_ext2_session_csv(fp)
        if d is None:
            continue
        aid=os.path.basename(fp).split("_")[0]
        tone_cls=classify_onset_evoked(d["z"], d["tone_flag"], d["tone_id"], d["time"],
                                       thr_z=p["thr_z"], consec_frames=p["consec_frames"], window_s=p["window_s"],
                                       early_tones=p["early_tones"], late_tones=p["late_tones"])
        for thr in thresholds:
            add_rows("Ext2", aid, d, tone_cls, thr)

    # Retrieval
    for npzpath in ret_npz:
        d=load_ret_npz(npzpath)
        aid=os.path.basename(npzpath).replace("_retrieval_zscored_traces.npz","")
        cmap=ret_maps.get(aid,{})
        tone_cls=[cmap.get(cid,"Neither") for cid in d["cell_ids"]]
        for thr in thresholds:
            add_rows("Retrieval", aid, d, tone_cls, thr)

    summary=pd.DataFrame(rows).sort_values(["session","freeze_resp_threshold","animal_id"])
    qc_df=pd.DataFrame(qc).drop_duplicates().sort_values(["session","animal_id"])

    summary.to_csv(os.path.join(outdir,"freezeResponsive_overlapToneOnset_byClass_perAnimal.csv"), index=False)
    qc_df.to_csv(os.path.join(outdir,"freezeResponsive_overlapToneOnset_QC_perAnimal.csv"), index=False)

    # plots: per session + threshold (animal mean ± SEM + dots)
    cats=[("EarlyOnly","p_EarlyOnly_within_freezeResp"),
          ("Overlap","p_Overlap_within_freezeResp"),
          ("LateOnly","p_LateOnly_within_freezeResp"),
          ("NotTone","p_notToneResp_within_freezeResp")]
    for thr in thresholds:
        for sess in ["Ext1","Ext2","Retrieval"]:
            df=summary[(summary["session"]==sess) & (summary["freeze_resp_threshold"]==thr)].copy()
            fig, ax=plt.subplots(figsize=(9.2,5.2))
            x=np.arange(len(cats))
            means=[]; sems=[]
            for i,(lab,col) in enumerate(cats):
                vals=df[col].astype(float).values
                vals=vals[np.isfinite(vals)]
                means.append(np.nanmean(vals) if len(vals)>0 else np.nan)
                sems.append(np.nanstd(vals, ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else np.nan)
                v2=df[["animal_id",col]].dropna()
                if len(v2)>0:
                    jitter=(rng.random(len(v2))-0.5)*0.16
                    ax.scatter(np.full(len(v2), i)+jitter, v2[col].astype(float).values, s=18, alpha=0.75)
            ax.bar(x, means, alpha=0.85)
            ax.errorbar(x, means, yerr=sems, fmt="none", capsize=4, linewidth=1.5, alpha=0.9)
            ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cats])
            ax.set_ylim(0,1.05)
            ax.set_ylabel("Proportion within freezing-responsive set")
            thr_lab="Δz≤0.0 (any decrease)" if thr==0.0 else "Δz≤-0.5 (strong decrease)"
            ax.set_title(f"{sess}: among freezing-responsive cells, overlap with tone-onset classes\n{thr_lab}")
            plt.tight_layout()
            outpng=os.path.join(outdir, f"{sess}_freezeRespOverlapToneResp_byClass_thr{str(thr).replace('.','p').replace('-','m')}_GROUPED_ANIMALMEAN_SEM.png")
            plt.savefig(outpng, dpi=300); plt.close()
            plot_files.append(outpng)

            # stacked mean composition
            comp={}
            for lab,col in cats:
                v=df[col].astype(float).values
                v=v[np.isfinite(v)]
                comp[lab]=np.nanmean(v) if len(v)>0 else 0.0
            fig, ax=plt.subplots(figsize=(6.2,3.0))
            bottom=0.0
            for lab,_ in cats:
                val=float(comp.get(lab,0.0))
                ax.bar([0],[val],bottom=bottom,alpha=0.9,label=lab)
                bottom+=val
            ax.set_ylim(0,1.05)
            ax.set_xticks([0]); ax.set_xticklabels([sess])
            ax.set_ylabel("Mean proportion within freezeResp")
            ax.set_title(f"{sess}: mean composition | thr {'Δz≤0.0' if thr==0.0 else 'Δz≤-0.5'}")
            ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5,-0.25))
            plt.tight_layout()
            outpng=os.path.join(outdir, f"{sess}_freezeRespOverlapToneResp_byClass_thr{str(thr).replace('.','p').replace('-','m')}_STACKED_ANIMALMEAN.png")
            plt.savefig(outpng, dpi=300, bbox_inches="tight"); plt.close()
            plot_files.append(outpng)

if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args.config)
    main(cfg)
