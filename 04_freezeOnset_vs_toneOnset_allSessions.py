import argparse, os
from utils_freezing import load_config
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    return ap.parse_args()

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from utils_freezing import (
    extract_npz_parts, extract_npz_from_zip, extract_csvs_from_zip, choose_one_ext2_csv_per_animal,
    load_ext1_npz, load_ret_npz, load_ext2_session_csv,
    load_ext1_class_maps, load_ret_class_maps, classify_onset_evoked, sex_from_animal_id,
    infer_fps, rising_edges, tone_only_freeze_onsets, compute_event_delta_mean, compute_event_peak_mean, ensure_dir, standardize_class_label
)

RESP={"EarlyOnly","Overlap","LateOnly"}

def main(cfg):
    outdir=cfg["paths"]["output_dir"]
    ensure_dir(outdir)

    sexmap=cfg.get("sex_mapping", {})
    ext1_maps=load_ext1_class_maps(cfg["paths"]["ext1_classes_zip"], os.path.join(outdir,"_ext1_classes"))
    ret_maps=load_ret_class_maps(cfg["paths"]["ret_classes_zip"], os.path.join(outdir,"_ret_classes"))

    ext1_npz=extract_npz_parts(cfg["paths"]["ext1_zscored_npz_parts"], outdir)
    ret_npz=extract_npz_from_zip(cfg["paths"]["ret_zscored_npz_zip"], outdir)
    ext2_csvs=extract_csvs_from_zip(cfg["paths"]["ext2_raw_zip"], os.path.join(outdir,"_ext2_raw"))
    ext2_files=choose_one_ext2_csv_per_animal(ext2_csvs)

    p=cfg["tone_onset_classification_ext2"]
    base_s=cfg["freezing_metrics"]["baseline_pre_s"]
    post_s=cfg["freezing_metrics"]["post_s"]
    peak_s=cfg["freezing_metrics"]["peak_window_s"]

    cell_rows=[]
    animal_rows=[]
    corr_rows=[]
    corr2_rows=[]

    sessions=[("Ext1", ext1_npz, "ext1"), ("Ext2", ext2_files, "ext2"), ("Retrieval", ret_npz, "ret")]
    for sess, paths, kind in sessions:
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

            sex=sex_from_animal_id(aid, sexmap)

            fps=infer_fps(d["time"])
            if not np.isfinite(fps): fps=10.0
            preF=int(round(base_s*fps))
            postF=int(round(post_s*fps))
            peakF=int(round(peak_s*fps))

            tone_onsets=rising_edges(d["tone_flag"])
            freeze_onsets=tone_only_freeze_onsets(d["tone_flag"], d["freeze"])

            tone_peak = compute_event_peak_mean(d["z"], tone_onsets, peakF)
            freeze_delta = compute_event_delta_mean(d["z"], freeze_onsets, preF, postF)
            freeze_peak = compute_event_peak_mean(d["z"], freeze_onsets, peakF)

            # cell table
            for cid, cls, tp, fd, fpv in zip(d["cell_ids"], tone_cls, tone_peak, freeze_delta, freeze_peak):
                cell_rows.append({
                    "session": sess,
                    "animal_id": aid,
                    "sex": sex,
                    "cell_id": str(cid),
                    "class": cls,
                    "tone_onset_peakMean_0to3s": float(tp) if np.isfinite(tp) else np.nan,
                    "freeze_onset_deltaMean_pre1s_post1s": float(fd) if np.isfinite(fd) else np.nan,
                    "freeze_onset_peakMean_0to3s": float(fpv) if np.isfinite(fpv) else np.nan,
                    "n_freeze_onsets_tone_used": int(len(freeze_onsets)),
                })

            # animal summary
            animal_rows.append({
                "session": sess,
                "animal_id": aid,
                "sex": sex,
                "fps": fps,
                "n_cells": int(d["z"].shape[1]),
                "n_tone_onsets": int(len(tone_onsets)),
                "n_freeze_onsets_in_tones": int(len(freeze_onsets)),
                "freeze_prop_during_tones": float(np.mean(d["freeze"][d["tone_flag"]==1]) if np.any(d["tone_flag"]==1) else np.nan),
                "mean_tone_onset_peakMean": float(np.nanmean(tone_peak)),
                "mean_freeze_onset_deltaMean": float(np.nanmean(freeze_delta)),
                "mean_freeze_onset_peakMean": float(np.nanmean(freeze_peak)),
            })

            # per-animal correlations (all cells + responsiveOnly)
            valid = np.isfinite(tone_peak) & np.isfinite(freeze_delta)
            n_used=int(np.sum(valid))
            pear_r=stats.pearsonr(tone_peak[valid], freeze_delta[valid]).statistic if n_used>=3 else np.nan
            pear_p=stats.pearsonr(tone_peak[valid], freeze_delta[valid]).pvalue if n_used>=3 else np.nan
            spear=stats.spearmanr(tone_peak[valid], freeze_delta[valid], nan_policy="omit")
            spe_r=float(spear.correlation) if n_used>=3 else np.nan
            spe_p=float(spear.pvalue) if n_used>=3 else np.nan

            resp_mask=np.isin(tone_cls, ["EarlyOnly","Overlap","LateOnly"])
            valid_r = valid & resp_mask
            n_used_r=int(np.sum(valid_r))
            pear_r_r=stats.pearsonr(tone_peak[valid_r], freeze_delta[valid_r]).statistic if n_used_r>=3 else np.nan
            pear_p_r=stats.pearsonr(tone_peak[valid_r], freeze_delta[valid_r]).pvalue if n_used_r>=3 else np.nan
            spear_r=stats.spearmanr(tone_peak[valid_r], freeze_delta[valid_r], nan_policy="omit")
            spe_r_r=float(spear_r.correlation) if n_used_r>=3 else np.nan
            spe_p_r=float(spear_r.pvalue) if n_used_r>=3 else np.nan

            corr_rows.append({
                "session": sess, "animal_id": aid, "sex": sex,
                "n_cells_used": n_used,
                "pearson_r": pear_r, "pearson_p": pear_p,
                "spearman_r": spe_r, "spearman_p": spe_p,
                "n_freeze_onsets_in_tones": int(len(freeze_onsets)),
                "n_cells_used_responsiveOnly": n_used_r,
                "pearson_r_responsiveOnly": pear_r_r, "pearson_p_responsiveOnly": pear_p_r,
                "spearman_r_responsiveOnly": spe_r_r, "spearman_p_responsiveOnly": spe_p_r,
            })

            # tonePeak vs freezePeak correlation (for plots)
            valid2=np.isfinite(tone_peak) & np.isfinite(freeze_peak)
            if np.sum(valid2)>=3:
                pear2=stats.pearsonr(tone_peak[valid2], freeze_peak[valid2]).statistic
            else:
                pear2=np.nan
            corr2_rows.append({
                "session": sess, "animal_id": aid, "sex": sex,
                "n_cells_used_delta": n_used,
                "pearson_r_tonePeak_vs_freezeDelta": pear_r,
                "pearson_r_tonePeak_vs_freezePeak": pear2,
            })

    cell_df=pd.DataFrame(cell_rows)
    animal_df=pd.DataFrame(animal_rows)
    corr_df=pd.DataFrame(corr_rows)
    corr2_df=pd.DataFrame(corr2_rows)

    cell_df.to_csv(os.path.join(outdir,"AllSessions_freezeOnset_vs_toneOnset_cellMetrics.csv"), index=False)
    animal_df.to_csv(os.path.join(outdir,"AllSessions_freezeOnset_vs_toneOnset_animalSummary.csv"), index=False)
    corr_df.to_csv(os.path.join(outdir,"AllSessions_freezeOnset_vs_toneOnset_cellCorrelation_perAnimal.csv"), index=False)
    corr2_df.to_csv(os.path.join(outdir,"AllSessions_freezeOnset_vs_toneOnset_correlations_deltaVsPeak_perAnimal.csv"), index=False)

    # plots by session with sex colors not enforced (matplotlib default cycle)
    def scatter_by_sex(df, xcol, ycol, title, outpng):
        fig, ax=plt.subplots(figsize=(6.2,5.2))
        for sex, sub in df.groupby("sex"):
            ax.scatter(sub[xcol], sub[ycol], s=22, alpha=0.8, label=sex)
        ax.axhline(0, alpha=0.2); ax.axvline(0, alpha=0.2)
        ax.set_xlabel(xcol); ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outpng, dpi=300)
        plt.close()

    for sess in ["Ext1","Ext2","Retrieval"]:
        sub=cell_df[cell_df["session"]==sess].copy()
        scatter_by_sex(
            sub, "tone_onset_peakMean_0to3s", "freeze_onset_deltaMean_pre1s_post1s",
            f"{sess}: tone onset peak vs freeze onset Δ (cells, by sex)",
            os.path.join(outdir, f"{sess}_corr_freezeOnsetDelta_vs_toneOnsetPeak_bySex.png")
        )
        scatter_by_sex(
            sub, "tone_onset_peakMean_0to3s", "freeze_onset_peakMean_0to3s",
            f"{sess}: tone onset peak vs freeze onset peak (cells, by sex)",
            os.path.join(outdir, f"{sess}_corr_tonePeak_vs_freezePeak_bySex.png")
        )

    # session comparison using animal means
    fig, ax=plt.subplots(figsize=(6.6,5.2))
    for sess, sub in animal_df.groupby("session"):
        ax.scatter(sub["mean_tone_onset_peakMean"], sub["mean_freeze_onset_deltaMean"], s=35, alpha=0.85, label=sess)
    ax.axhline(0, alpha=0.2); ax.axvline(0, alpha=0.2)
    ax.set_xlabel("Animal mean tone onset peakMean (0..3s)")
    ax.set_ylabel("Animal mean freeze onset Δ (post1s-pre1s)")
    ax.set_title("All sessions: animal means")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "AllSessions_animalMeans_scatter_freezeOnsetDelta_vs_toneOnsetPeak.png"), dpi=300)
    plt.close()

    # correlations per animal across sessions
    fig, ax=plt.subplots(figsize=(7.2,4.8))
    for sess, sub in corr2_df.groupby("session"):
        ax.scatter(np.full(len(sub), sess), sub["pearson_r_tonePeak_vs_freezeDelta"], s=45, alpha=0.85, label=sess)
    ax.set_ylabel("Pearson r (tonePeak vs freezeΔ)")
    ax.set_title("Per-animal correlations across sessions")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "AllSessions_corr_freezeOnsetDelta_vs_toneOnsetPeak_sessionComparison.png"), dpi=300)
    plt.close()

if __name__=="__main__":
    args=parse_args()
    cfg=load_config(args.config)
    main(cfg)
