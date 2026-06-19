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
    infer_fps, tone_only_freeze_onsets, event_triggered_traces, ensure_dir
)

def main(cfg):
    outdir=cfg["paths"]["output_dir"]
    ensure_dir(outdir)

    # Extract inputs
    ext1_npz=extract_npz_parts(cfg["paths"]["ext1_zscored_npz_parts"], outdir)
    ret_npz=extract_npz_from_zip(cfg["paths"]["ret_zscored_npz_zip"], outdir)
    ext2_csvs=extract_csvs_from_zip(cfg["paths"]["ext2_raw_zip"], os.path.join(outdir,"_ext2_raw"))
    ext2_files=choose_one_ext2_csv_per_animal(ext2_csvs)

    pre_s=cfg["freezing_metrics"]["trace_window_pre_s"]
    post_s=cfg["freezing_metrics"]["trace_window_post_s"]
    base_s=cfg["freezing_metrics"]["baseline_pre_s"]

    sessions=[("Ext1", ext1_npz, "ext1"), ("Ext2", ext2_files, "ext2"), ("Retrieval", ret_npz, "ret")]
    counts=[]

    for sess, paths, kind in sessions:
        # pooled cell-weighted
        pooled_cells=[]
        pooled_animals=[]  # per animal mean trace (for animal-weighted)

        for p in paths:
            if kind=="ext1":
                d=load_ext1_npz(p)
            elif kind=="ret":
                d=load_ret_npz(p)
            else:
                d=load_ext2_session_csv(p)
                if d is None:
                    continue

            fps=infer_fps(d["time"]); 
            if not np.isfinite(fps): fps=10.0
            preF=int(round(pre_s*fps)); postF=int(round(post_s*fps)); baseF=int(round(base_s*fps))
            t_axis=np.arange(-preF, postF+1)/fps

            # tone-only freeze onsets
            onsets=tone_only_freeze_onsets(d["tone_flag"], d["freeze"])
            traces=event_triggered_traces(d["z"], onsets, preF, postF, baseline_pre_frames=baseF)  # n_ev x C x T
            counts.append({"session":sess, "animal_id":os.path.basename(p).split("_")[0], "n_freeze_onsets_tone_used":int(traces.shape[0]), "fps":fps})
            if traces.shape[0]==0:
                continue

            # per cell: mean over events
            cell_mean=np.nanmean(traces, axis=0)  # C x T
            pooled_cells.append((t_axis, cell_mean))
            pooled_animals.append((t_axis, np.nanmean(cell_mean, axis=0)))  # T

        # build common time grid by interpolation if mixed fps across animals
        # (use 0.1s grid for stability)
        COMMON=np.arange(-pre_s, post_s+1e-9, 0.1)
        # cell-weighted
        if len(pooled_cells)>0:
            # interpolate each cell's trace to COMMON time grid
            Cmat=np.vstack([
                np.vstack([np.interp(COMMON, t_ax, cm[i]) for i in range(cm.shape[0])])
                for (t_ax, cm) in pooled_cells
            ])  # (sumC) x T
            mean=np.nanmean(Cmat, axis=0)
            sem=np.nanstd(Cmat, axis=0, ddof=1)/np.sqrt(Cmat.shape[0]) if Cmat.shape[0]>1 else np.full_like(mean,np.nan)
            pd.DataFrame({"time_s":COMMON, "mean":mean, "sem":sem, "n_cells":np.full_like(COMMON, Cmat.shape[0], dtype=int)}).to_csv(
                os.path.join(outdir, f"{sess}_freezeOnset_deltaZ_meanTrace_-5to5_CELLWEIGHTED.csv"), index=False
            )
            fig, ax=plt.subplots(figsize=(7.5,4.5))
            ax.plot(COMMON, mean, linewidth=2)
            ax.fill_between(COMMON, mean-sem, mean+sem, alpha=0.2)
            ax.axvline(0, alpha=0.6)
            ax.axhline(0, alpha=0.25)
            ax.set_xlabel("Time from freezing onset (s)")
            ax.set_ylabel("Δz (baseline: -1..0 s)")
            ax.set_title(f"{sess} freezing-onset Δz mean trace (cell-weighted)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{sess}_freezeOnset_deltaZ_meanTrace_-5to5_CELLWEIGHTED.png"), dpi=300)
            plt.close()

        # animal-weighted
        if len(pooled_animals)>0:
            Amat=np.vstack([np.interp(COMMON, t_ax, a) for (t_ax, a) in pooled_animals])
            mean=np.nanmean(Amat, axis=0)
            sem=np.nanstd(Amat, axis=0, ddof=1)/np.sqrt(Amat.shape[0]) if Amat.shape[0]>1 else np.full_like(mean,np.nan)
            pd.DataFrame({"time_s":COMMON, "mean":mean, "sem":sem, "n_animals":np.full_like(COMMON, Amat.shape[0], dtype=int)}).to_csv(
                os.path.join(outdir, f"{sess}_freezeOnset_deltaZ_meanTrace_-5to5_ANIMALWEIGHTED.csv"), index=False
            )
            fig, ax=plt.subplots(figsize=(7.5,4.5))
            ax.plot(COMMON, mean, linewidth=2)
            ax.fill_between(COMMON, mean-sem, mean+sem, alpha=0.2)
            ax.axvline(0, alpha=0.6)
            ax.axhline(0, alpha=0.25)
            ax.set_xlabel("Time from freezing onset (s)")
            ax.set_ylabel("Δz (baseline: -1..0 s)")
            ax.set_title(f"{sess} freezing-onset Δz mean trace (animal-weighted)")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{sess}_freezeOnset_deltaZ_meanTrace_-5to5_ANIMALWEIGHTED.png"), dpi=300)
            plt.close()

    # event counts
    pd.DataFrame(counts).to_csv(os.path.join(outdir, "freezeOnset_deltaZ_trace_eventCounts_perAnimal.csv"), index=False)

if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args.config)
    main(cfg)
