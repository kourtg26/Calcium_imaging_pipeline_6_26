import argparse, os
import numpy as np, pandas as pd
from utils_freezing import (
    load_config, ensure_dir,
    extract_npz_parts, extract_csvs_from_zip, choose_one_ext2_csv_per_animal,
    load_ext1_npz, load_ext2_session_csv,
    infer_fps, rising_edges, tone_only_freeze_onsets,
    compute_event_delta_mean
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    return ap.parse_args()

# Tone bins
TONE_BINS = [
    (1, 3, "T1-3"),
    (4, 6, "T4-6"),
    (7, 9, "T7-9"),
    (10, 12, "T10-12"),
]

def tone_bin_label(tid: int) -> str | None:
    for lo, hi, lab in TONE_BINS:
        if lo <= tid <= hi:
            return lab
    return None

def compute_event_deltas(z: np.ndarray, onsets: np.ndarray, pre_frames: int, post_frames: int) -> np.ndarray:
    """Return per-event per-cell delta (mean post - mean pre), shape (n_events, n_cells)."""
    if len(onsets) == 0:
        return np.zeros((0, z.shape[1]), dtype=float)
    deltas = []
    for idx in onsets:
        if idx - pre_frames < 0 or idx + post_frames > z.shape[0]:
            continue
        base = np.nanmean(z[idx - pre_frames:idx, :], axis=0)
        post = np.nanmean(z[idx:idx + post_frames, :], axis=0)
        deltas.append(post - base)
    if len(deltas) == 0:
        return np.zeros((0, z.shape[1]), dtype=float)
    return np.vstack(deltas)

def analyze_session(session: str, datasets: list, freeze_thr: float):
    rows = []
    per_animal = []
    for aid, d in datasets:
        fps = infer_fps(d["time"])
        if not np.isfinite(fps):
            fps = 10.0
        preF = int(round(cfg["freezing_metrics"]["baseline_pre_s"] * fps))
        postF = int(round(cfg["freezing_metrics"]["post_s"] * fps))

        # freeze-responsive mask (mean delta across tone-only freeze onsets)
        freeze_on = tone_only_freeze_onsets(d["tone_flag"], d["freeze"])
        delta_mean = compute_event_delta_mean(d["z"], freeze_on, preF, postF)
        freeze_resp = np.isfinite(delta_mean) & (delta_mean <= freeze_thr)

        # Option 1: most negative tone-evoked delta bin (within tone onsets)
        tone_onsets = rising_edges(d["tone_flag"])
        tone_deltas = compute_event_deltas(d["z"], tone_onsets, preF, postF)
        tone_bins = [tone_bin_label(int(d["tone_id"][i])) if i < len(d["tone_id"]) else None for i in tone_onsets]

        # Option 2: most negative freeze-onset delta bin (tone-only freeze onsets)
        freeze_deltas = compute_event_deltas(d["z"], freeze_on, preF, postF)
        freeze_bins = [tone_bin_label(int(d["tone_id"][i])) if i < len(d["tone_id"]) else None for i in freeze_on]

        # Initialize counts
        counts_tone = {lab: 0 for _, _, lab in TONE_BINS}
        counts_freeze = {lab: 0 for _, _, lab in TONE_BINS}

        # For each freeze-responsive cell, assign bin from tone-evoked deltas (Option 1)
        if tone_deltas.shape[0] > 0:
            tone_deltas = tone_deltas[:, freeze_resp]  # events x resp_cells
            for ci in range(tone_deltas.shape[1]):
                # mean delta per bin
                bin_means = {}
                for lab in counts_tone:
                    idxs = [k for k, b in enumerate(tone_bins) if b == lab]
                    if not idxs:
                        continue
                    bin_means[lab] = np.nanmean(tone_deltas[idxs, ci])
                if bin_means:
                    # most negative (minimum) mean delta
                    best_lab = min(bin_means, key=lambda k: bin_means[k])
                    counts_tone[best_lab] += 1

        # For each freeze-responsive cell, assign bin from freeze-onset deltas (Option 2)
        if freeze_deltas.shape[0] > 0:
            freeze_deltas = freeze_deltas[:, freeze_resp]  # events x resp_cells
            for ci in range(freeze_deltas.shape[1]):
                # choose event with most negative delta
                ev_idx = int(np.nanargmin(freeze_deltas[:, ci]))
                lab = freeze_bins[ev_idx] if ev_idx < len(freeze_bins) else None
                if lab in counts_freeze:
                    counts_freeze[lab] += 1

        # per-animal rows
        n_resp = int(np.sum(freeze_resp))
        for lab in counts_tone:
            per_animal.append({
                "session": session,
                "animal_id": aid,
                "method": "tone_evoked_min_delta_bin",
                "tone_bin": lab,
                "count": counts_tone[lab],
                "n_freeze_resp": n_resp
            })
            per_animal.append({
                "session": session,
                "animal_id": aid,
                "method": "freeze_onset_min_delta_bin",
                "tone_bin": lab,
                "count": counts_freeze[lab],
                "n_freeze_resp": n_resp
            })

        # summary rows (aggregate per session later)
        rows.append((counts_tone, counts_freeze))

    # Aggregate per session
    summary = []
    if rows:
        # sum counts across animals
        total_tone = {lab: 0 for _, _, lab in TONE_BINS}
        total_freeze = {lab: 0 for _, _, lab in TONE_BINS}
        for ct, cf in rows:
            for lab in total_tone:
                total_tone[lab] += ct.get(lab, 0)
                total_freeze[lab] += cf.get(lab, 0)
        # compute proportions
        tot_t = sum(total_tone.values())
        tot_f = sum(total_freeze.values())
        for lab in total_tone:
            summary.append({
                "session": session,
                "method": "tone_evoked_min_delta_bin",
                "tone_bin": lab,
                "count": total_tone[lab],
                "prop": (total_tone[lab] / tot_t) if tot_t > 0 else np.nan
            })
            summary.append({
                "session": session,
                "method": "freeze_onset_min_delta_bin",
                "tone_bin": lab,
                "count": total_freeze[lab],
                "prop": (total_freeze[lab] / tot_f) if tot_f > 0 else np.nan
            })

    return per_animal, summary

def main(cfg):
    outdir = cfg["paths"]["output_dir"]
    ensure_dir(outdir)

    freeze_thr = -0.25

    # Ext1 datasets
    ext1_npz = extract_npz_parts(cfg["paths"]["ext1_zscored_npz_parts"], outdir)
    ext1_data = []
    for p in ext1_npz:
        d = load_ext1_npz(p)
        aid = os.path.basename(p).replace("_ext1_zscored_traces.npz", "")
        ext1_data.append((aid, d))

    # Ext2 datasets
    ext2_csvs = extract_csvs_from_zip(cfg["paths"]["ext2_raw_zip"], os.path.join(outdir, "_ext2_raw"))
    ext2_files = choose_one_ext2_csv_per_animal(ext2_csvs)
    ext2_data = []
    for fp in ext2_files:
        d = load_ext2_session_csv(fp)
        if d is None:
            continue
        aid = os.path.basename(fp).split("_")[0]
        ext2_data.append((aid, d))

    per_animal_all = []
    summary_all = []
    pa, summ = analyze_session("Ext1", ext1_data, freeze_thr)
    per_animal_all += pa
    summary_all += summ
    pa, summ = analyze_session("Ext2", ext2_data, freeze_thr)
    per_animal_all += pa
    summary_all += summ

    per_animal_df = pd.DataFrame(per_animal_all)
    if not per_animal_df.empty:
        per_animal_df.to_csv(os.path.join(outdir, "freezeResp_toneBin_assignment_perAnimal.csv"), index=False)
    summary_df = pd.DataFrame(summary_all)
    if not summary_df.empty:
        summary_df.to_csv(os.path.join(outdir, "freezeResp_toneBin_assignment_summary.csv"), index=False)

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    main(cfg)
