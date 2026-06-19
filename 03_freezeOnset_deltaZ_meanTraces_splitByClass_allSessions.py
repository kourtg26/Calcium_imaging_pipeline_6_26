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
    infer_fps, tone_only_freeze_onsets, event_triggered_traces, ensure_dir, standardize_class_label
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

    pre_s=cfg["freezing_metrics"]["trace_window_pre_s"]
    post_s=cfg["freezing_metrics"]["trace_window_post_s"]
    base_s=cfg["freezing_metrics"]["baseline_pre_s"]
    p=cfg["tone_onset_classification_ext2"]

    sessions=[("Ext1", ext1_npz, "ext1"), ("Ext2", ext2_files, "ext2"), ("Retrieval", ret_npz, "ret")]
    counts=[]
    summary_counts=[]

    for sess, paths, kind in sessions:
        pooled_cells={c: [] for c in CLASSES}
        pooled_animals={c: [] for c in CLASSES}
        time_grid=None

        for pth in paths:
            if kind=="ext1":
                d=load_ext1_npz(pth)
                aid=os.path.basename(pth).replace("_ext1_zscored_traces.npz","")
                cmap=ext1_maps.get(aid,{})
                tone_cls=np.array([standardize_class_label(cmap.get(cid,"Neither")) for cid in d["cell_ids"]], dtype=object)
            elif kind=="ret":
                d=load_ret_npz(pth)
                aid=os.path.basename(pth).replace("_retrieval_zscored_traces.npz","")
                cmap=ret_maps.get(aid,{})
                tone_cls=np.array([standardize_class_label(cmap.get(cid,"Neither")) for cid in d["cell_ids"]], dtype=object)
            else:
                d=load_ext2_session_csv(pth)
                if d is None:
                    continue
                aid=os.path.basename(pth).split("_")[0]
                tone_cls=classify_onset_evoked(d["z"], d["tone_flag"], d["tone_id"], d["time"],
                                               thr_z=p["thr_z"], consec_frames=p["consec_frames"], window_s=p["window_s"],
                                               early_tones=p["early_tones"], late_tones=p["late_tones"])
            fps=infer_fps(d["time"]); 
            if not np.isfinite(fps): fps=10.0
            preF=int(round(pre_s*fps)); postF=int(round(post_s*fps)); baseF=int(round(base_s*fps))
            t_axis=np.arange(-preF, postF+1)/fps
            if time_grid is None:
                time_grid=t_axis

            onsets=tone_only_freeze_onsets(d["tone_flag"], d["freeze"])
            traces=event_triggered_traces(d["z"], onsets, preF, postF, baseline_pre_frames=baseF)
            counts.append({"session":sess, "animal_id":aid, "n_freeze_onsets_tone_used":int(traces.shape[0]), "fps":fps})

            if traces.shape[0]==0:
                continue
            # per cell mean over events
            cell_mean=np.nanmean(traces, axis=0)  # C x T

            for c in CLASSES:
                mask=(tone_cls==c)
                summary_counts.append({"session":sess, "animal_id":aid, "class":c, "n_cells":int(np.sum(mask))})
                if not np.any(mask):
                    continue
                cm=cell_mean[mask, :]  # cells x T
                pooled_cells[c].append(cm)
                pooled_animals[c].append(np.nanmean(cm, axis=0))

        COMMON=np.arange(-pre_s, post_s+1e-9, 0.1)

        # cell-weighted per class
        for c in CLASSES:
            if len(pooled_cells[c])>0:
                M=np.vstack([np.interp(COMMON, time_grid, cm.T).T for cm in pooled_cells[c]])  # pooled cells x T
                mean=np.nanmean(M, axis=0)
                sem=np.nanstd(M, axis=0, ddof=1)/np.sqrt(M.shape[0]) if M.shape[0]>1 else np.full_like(mean,np.nan)
                pd.DataFrame({"time_s":COMMON, "mean":mean, "sem":sem, "n_cells":np.full_like(COMMON, M.shape[0], dtype=int)}).to_csv(
                    os.path.join(outdir, f"{sess}_freezeOnset_deltaZ_-5to5_{c}_CELLWEIGHTED.csv"), index=False
                )

        # animal-weighted per class
        for c in CLASSES:
            if len(pooled_animals[c])>0:
                A=np.vstack([np.interp(COMMON, time_grid, a) for a in pooled_animals[c]])
                mean=np.nanmean(A, axis=0)
                sem=np.nanstd(A, axis=0, ddof=1)/np.sqrt(A.shape[0]) if A.shape[0]>1 else np.full_like(mean,np.nan)
                pd.DataFrame({"time_s":COMMON, "mean":mean, "sem":sem, "n_animals":np.full_like(COMMON, A.shape[0], dtype=int)}).to_csv(
                    os.path.join(outdir, f"{sess}_freezeOnset_deltaZ_-5to5_{c}_ANIMALWEIGHTED.csv"), index=False
                )

        # make summary plots (overlay 3 classes)
        def overlay_plot(weight, pooled_dict, tag):
            fig, ax=plt.subplots(figsize=(8.5,5.0))
            for c in CLASSES:
                if len(pooled_dict[c])==0:
                    continue
                if weight=="cell":
                    M=np.vstack([np.interp(COMMON, time_grid, cm.T).T for cm in pooled_dict[c]])
                    mean=np.nanmean(M, axis=0)
                    sem=np.nanstd(M, axis=0, ddof=1)/np.sqrt(M.shape[0]) if M.shape[0]>1 else np.full_like(mean,np.nan)
                    label=f"{c} (cells={M.shape[0]})"
                else:
                    A=np.vstack([np.interp(COMMON, time_grid, a) for a in pooled_dict[c]])
                    mean=np.nanmean(A, axis=0)
                    sem=np.nanstd(A, axis=0, ddof=1)/np.sqrt(A.shape[0]) if A.shape[0]>1 else np.full_like(mean,np.nan)
                    label=f"{c} (animals={A.shape[0]})"
                ax.plot(COMMON, mean, linewidth=2, label=label)
                ax.fill_between(COMMON, mean-sem, mean+sem, alpha=0.18)
            ax.axvline(0, alpha=0.6); ax.axhline(0, alpha=0.25)
            ax.set_xlabel("Time from freezing onset (s)")
            ax.set_ylabel("Δz (baseline: -1..0 s)")
            ax.set_title(f"{sess} freezing-onset Δz split by tone-onset class ({tag})")
            ax.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{sess}_freezeOnset_deltaZ_-5to5_splitByClass_{tag}.png"), dpi=300)
            plt.close()

        overlay_plot("cell", pooled_cells, "CELLWEIGHTED")
        overlay_plot("animal", pooled_animals, "ANIMALWEIGHTED")

    pd.DataFrame(counts).to_csv(os.path.join(outdir, "freezeOnset_deltaZ_splitByClass_eventCounts_perAnimal.csv"), index=False)
    pd.DataFrame(summary_counts).to_csv(os.path.join(outdir, "freezeOnset_deltaZ_splitByClass_summaryCounts.csv"), index=False)

if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args.config)
    main(cfg)
