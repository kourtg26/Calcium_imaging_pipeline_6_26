import os, glob, zipfile, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR="/mnt/data"
work=os.path.join(DATA_DIR,"_ext1_meantrace_work")  # reuse prior extraction dir
if not os.path.exists(work):
    os.makedirs(work, exist_ok=True)

# Ensure required archives are extracted (idempotent)
zips = [
    os.path.join(DATA_DIR,"ext1_classes_proportions_only.zip"),
    os.path.join(DATA_DIR,"ext1_zscored_npz_part1.zip"),
    os.path.join(DATA_DIR,"ext1_zscored_npz_part2.zip"),
    os.path.join(DATA_DIR,"ext1_zscored_npz_part3.zip"),
]
for zp in zips:
    if not os.path.exists(zp):
        raise FileNotFoundError(f"Missing required zip: {zp}")
    with zipfile.ZipFile(zp,"r") as zf:
        zf.extractall(work)

class_csvs = sorted(glob.glob(os.path.join(work,"*_ext1_onset_evoked_cell_classes.csv")))
if not class_csvs:
    class_csvs = sorted(glob.glob(os.path.join(work,"*cell_classes*.csv")))
npzs = sorted(glob.glob(os.path.join(work,"*.npz")))

animal_class={}
for fp in class_csvs:
    animal=os.path.basename(fp).split("_")[0]
    df=pd.read_csv(fp)
    if "class" not in df.columns and "group" in df.columns:
        df=df.rename(columns={"group":"class"})
    if "cell_id" not in df.columns:
        for c in df.columns:
            if "cell" in c.lower():
                df=df.rename(columns={c:"cell_id"})
                break
    animal_class[animal]=df

animal_npz={}
for fp in npzs:
    base=os.path.basename(fp)
    if base.endswith("_ext1_zscored_traces.npz") or "_ext1_" in base:
        animal=base.split("_")[0]
    else:
        animal=base.split(".")[0].split("_")[0]
    # choose the expected ext1 zscored trace file if multiple
    if animal in animal_npz:
        # prefer *_ext1_zscored_traces.npz
        if base.endswith("_ext1_zscored_traces.npz"):
            animal_npz[animal]=fp
    else:
        animal_npz[animal]=fp

animals=sorted(set(animal_class.keys()).intersection(animal_npz.keys()))
if not animals:
    raise RuntimeError("No overlapping animals between class CSVs and NPZs.")

def tone_onsets_from_flag(flag):
    f=np.asarray(flag).astype(int)
    on=np.where((f[1:]==1) & (f[:-1]==0))[0]+1
    if f[0]==1:
        on=np.r_[0,on]
    return on.tolist()

def interp_cell_trace_for_onset(t, z, onset_idx, win_s, target_t):
    t0=t[onset_idx]
    mask=(t>=t0-win_s) & (t<=t0+win_s)
    if mask.sum()<3:
        return np.full_like(target_t, np.nan, dtype=float)
    rel=t[mask]-t0
    vals=z[mask]
    finite=np.isfinite(rel) & np.isfinite(vals)
    if finite.sum()<3:
        return np.full_like(target_t, np.nan, dtype=float)
    rel_f=rel[finite]; v_f=vals[finite]
    order=np.argsort(rel_f)
    rel_f=rel_f[order]; v_f=v_f[order]
    out=np.interp(target_t, rel_f, v_f)
    out[(target_t<rel_f[0]) | (target_t>rel_f[-1])] = np.nan
    return out

def nansem(a, axis=0):
    a=np.asarray(a,float)
    n=np.sum(np.isfinite(a), axis=axis)
    sd=np.nanstd(a, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem=sd/np.sqrt(n)
    return sem, n

# common target grid
dts=[]
for animal in animals:
    npz=np.load(animal_npz[animal])
    t=npz["Time_s"].astype(float)
    tt=t[np.isfinite(t)]
    dt=float(np.median(np.diff(tt)))
    dts.append(dt)
dt_target=float(np.median(dts))
target_t=np.arange(-5.0, 5.0 + dt_target/2, dt_target)

# label mapping
repl={
    "early_only":"EarlyOnly",
    "late_only":"LateOnly",
    "overlap":"Overlap",
    "neither":"Neither",
    "EarlyOnly":"EarlyOnly",
    "LateOnly":"LateOnly",
    "Overlap":"Overlap",
    "Neither":"Neither"
}

# Pooling structure: condition -> group -> list of per-cell traces
pooled = {
    "EarlyTones": {"EarlyCells": [], "LateCells": []},
    "LateTones":  {"EarlyCells": [], "LateCells": []},
}
cell_counts = {cond:{grp:0 for grp in pooled[cond]} for cond in pooled}
animals_used=set()

for animal in animals:
    cls=animal_class[animal].copy()
    cls["class_norm"]=cls["class"].astype(str).map(lambda x: repl.get(x, x))
    npz=np.load(animal_npz[animal], allow_pickle=True)
    t=npz["Time_s"].astype(float)
    Z=npz["Z"].astype(float)  # T x N
    cell_ids=[str(x) for x in npz["cell_ids"]]
    if "ToneFlag" in npz.files:
        toneflag=npz["ToneFlag"].astype(int)
    elif "ToneIndex" in npz.files:
        toneflag=(npz["ToneIndex"].astype(int)>0).astype(int)
    else:
        continue
    
    onsets=tone_onsets_from_flag(toneflag)
    if len(onsets)<6:
        continue
    early_onsets=onsets[:3]
    late_onsets=onsets[-3:]
    idx_map={cid:i for i,cid in enumerate(cell_ids)}
    
    early_cells=[idx_map[c] for c in cls.loc[cls["class_norm"]=="EarlyOnly","cell_id"].astype(str) if c in idx_map]
    late_cells =[idx_map[c] for c in cls.loc[cls["class_norm"]=="LateOnly","cell_id"].astype(str) if c in idx_map]
    
    def add_cells(cell_idx_list, onset_list, cond_key, grp_key):
        if len(cell_idx_list)==0:
            return
        for j in cell_idx_list:
            per_onset=[]
            for oi in onset_list:
                per_onset.append(interp_cell_trace_for_onset(t, Z[:,j], oi, 5.0, target_t))
            per_onset=np.vstack(per_onset)
            cell_trace=np.nanmean(per_onset, axis=0)  # average across tones, per cell
            pooled[cond_key][grp_key].append(cell_trace)
        cell_counts[cond_key][grp_key]+=len(cell_idx_list)

    # Early tones
    add_cells(early_cells, early_onsets, "EarlyTones", "EarlyCells")
    add_cells(late_cells,  early_onsets, "EarlyTones", "LateCells")
    # Late tones
    add_cells(early_cells, late_onsets,  "LateTones", "EarlyCells")
    add_cells(late_cells,  late_onsets,  "LateTones", "LateCells")
    
    animals_used.add(animal)

# Compute stats and output
out_files=[]
meta={
    "dt_target": dt_target,
    "target_t": target_t.tolist(),
    "animals_used": sorted(list(animals_used)),
    "cell_counts": cell_counts,
    "notes": "Cell-weighted mean: per-cell mean trace (averaged across early/late tones), then cohort mean+SEM across pooled cells (cells as n)."
}

def stats_from_traces(traces_list):
    if not traces_list:
        return None
    A=np.vstack(traces_list)
    mean=np.nanmean(A, axis=0)
    sem, n=nansem(A, axis=0)
    return mean, sem, n, A.shape[0]

def plot_two(condition, stats_E, stats_L, out_png):
    plt.figure(figsize=(6.5,4.0))
    if stats_E is not None:
        m,s,n, n_cells = stats_E
        plt.plot(target_t, m, label=f"Early-responding cells (n cells={n_cells})")
        plt.fill_between(target_t, m-s, m+s, alpha=0.25)
    if stats_L is not None:
        m,s,n, n_cells = stats_L
        plt.plot(target_t, m, label=f"Late-responding cells (n cells={n_cells})")
        plt.fill_between(target_t, m-s, m+s, alpha=0.25)
    plt.axvline(0, linewidth=1)
    plt.xlim(-5,5)
    plt.xlabel("Time from tone onset (s)")
    plt.ylabel("Z-scored activity")
    plt.title(f"{condition}: mean ± SEM (cell-weighted)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def write_csv(condition, stats_E, stats_L, out_csv):
    df=pd.DataFrame({"time_s": target_t})
    if stats_E is not None:
        m,s,n, n_cells = stats_E
        df["EarlyCells_mean"]=m
        df["EarlyCells_sem"]=s
        df["EarlyCells_nCells"]=n
    if stats_L is not None:
        m,s,n, n_cells = stats_L
        df["LateCells_mean"]=m
        df["LateCells_sem"]=s
        df["LateCells_nCells"]=n
    df.to_csv(out_csv, index=False)

for cond in ["EarlyTones","LateTones"]:
    stats_E=stats_from_traces(pooled[cond]["EarlyCells"])
    stats_L=stats_from_traces(pooled[cond]["LateCells"])
    png=os.path.join(DATA_DIR, f"Ext1_meanTrace_{cond}_window-5to5_meanSEM_cellWeighted.png")
    csv=os.path.join(DATA_DIR, f"Ext1_meanTrace_{cond}_window-5to5_meanSEM_cellWeighted.csv")
    plot_two(cond, stats_E, stats_L, png)
    write_csv(cond, stats_E, stats_L, csv)
    out_files.extend([png,csv])

meta_path=os.path.join(DATA_DIR,"Ext1_meanTrace_cellWeighted_run_metadata.json")
with open(meta_path,"w") as f:
    json.dump(meta,f,indent=2)
out_files.append(meta_path)

bundle=os.path.join(DATA_DIR,"ext1_meanTrace_plots_-5to5_meanSEM_cellWeighted_bundle.zip")
with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as zf:
    for fp in out_files:
        if os.path.exists(fp):
            zf.write(fp, arcname=os.path.basename(fp))

bundle, out_files

