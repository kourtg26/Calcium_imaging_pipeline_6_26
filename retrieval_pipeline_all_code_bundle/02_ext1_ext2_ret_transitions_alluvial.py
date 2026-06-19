#!/usr/bin/env python3
"""
02_ext1_ext2_ret_transitions_alluvial.py

Builds Ext1 → Ext2 → Retrieval transition tables and plots.

Inputs:
  - ext1_classes_proportions_only.zip
  - ext2_classes_proportions_only_UPDATED10489.zip
  - ret_classes_proportions_only.zip

Key choices:
  - Cell IDs are normalized (extract trailing digits, pad to 3): undecidedC12 -> C012
  - Class labels are normalized (e.g., late_only -> LateOnly)
  - Cells are INNER JOINED across Ext1, Ext2, Ret within animal:
      -> only cells present in ALL 3 sessions contribute
  - Outputs include static alluvial (SVG/PDF/PNG) + interactive plotly sankey HTML.
"""
import os, re, glob, zipfile, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path as MplPath
import plotly.graph_objects as go

CLASSES=["EarlyOnly","Overlap","LateOnly","Neither"]

def load_config(cfg_path):
    with open(cfg_path,"r") as f:
        return json.load(f)

def sex_of(animal_id, female_set, male_set):
    a=str(animal_id)
    if a in female_set: return "Female"
    if a in male_set: return "Male"
    return "Unknown"

def canonical_animal_from_fname(fname: str) -> str:
    b=os.path.basename(fname)
    m=re.search(r"animal(\d+)", b, flags=re.IGNORECASE)
    if m: return m.group(1)
    m=re.search(r"(\d{1,5})", b)
    if m: return m.group(1)
    return b.split("_")[0]

def normalize_cell_id_any(cid: str, width: int = 3) -> str:
    s=str(cid).strip()
    ms=re.findall(r"(\d+)", s)
    if ms:
        return "C"+ms[-1].zfill(width)
    return s

def normalize_class_label(lbl: str) -> str:
    s=str(lbl).strip()
    if s in CLASSES:
        return s
    s2=re.sub(r"[\s\-]+","_", s.lower())
    if s2.startswith("early"):
        return "EarlyOnly"
    if s2.startswith("late"):
        return "LateOnly"
    if "overlap" in s2:
        return "Overlap"
    if "neither" in s2 or s2 in ("none","na","nan","no"):
        return "Neither"
    return s

def load_classes_from_extracted(dirpath, pattern):
    files=sorted(glob.glob(os.path.join(dirpath, pattern)))
    out={}
    for fp in files:
        animal=canonical_animal_from_fname(fp)
        df=pd.read_csv(fp)
        df.columns=[c.strip() for c in df.columns]
        if "cell_id" not in df.columns:
            df=df.rename(columns={df.columns[0]:"cell_id"})
        if "class" not in df.columns:
            for c in df.columns:
                if c.lower()=="class":
                    df=df.rename(columns={c:"class"})
                    break
        if "cell_id" not in df.columns or "class" not in df.columns:
            continue
        d=df[["cell_id","class"]].copy()
        d["cell_id"]=d["cell_id"].map(lambda x: normalize_cell_id_any(x,3))
        d["class"]=d["class"].map(normalize_class_label)
        d=d[d["class"].isin(CLASSES)]
        out[animal]=d
    return out

def make_plotly_sankey(df, title, out_html):
    if len(df)==0:
        return None
    nodes=[]
    node_index={}
    for stage in ["Ext1","Ext2","Ret"]:
        for c in CLASSES:
            lab=f"{stage} {c}"
            node_index[(stage,c)]=len(nodes)
            nodes.append(lab)

    g12=df.groupby(["ext1_class","ext2_class"]).size().reset_index(name="value")
    g2r=df.groupby(["ext2_class","ret_class"]).size().reset_index(name="value")

    sources=[]; targets=[]; values=[]
    for _,r in g12.iterrows():
        sources.append(node_index[("Ext1", r["ext1_class"])])
        targets.append(node_index[("Ext2", r["ext2_class"])])
        values.append(int(r["value"]))
    for _,r in g2r.iterrows():
        sources.append(node_index[("Ext2", r["ext2_class"])])
        targets.append(node_index[("Ret", r["ret_class"])])
        values.append(int(r["value"]))

    fig=go.Figure(data=[go.Sankey(
        node=dict(label=nodes, pad=15, thickness=15),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title_text=title, font_size=12)
    fig.write_html(out_html)
    return out_html

def alluvial_three_stage(df, title, out_prefix, out_dir):
    total=int(len(df))
    if total==0:
        return None

    s1=df["ext1_class"].value_counts().reindex(CLASSES, fill_value=0)
    s2=df["ext2_class"].value_counts().reindex(CLASSES, fill_value=0)
    s3=df["ret_class"].value_counts().reindex(CLASSES, fill_value=0)

    l12=df.groupby(["ext1_class","ext2_class"]).size().reindex(pd.MultiIndex.from_product([CLASSES,CLASSES]), fill_value=0)
    l2r=df.groupby(["ext2_class","ret_class"]).size().reindex(pd.MultiIndex.from_product([CLASSES,CLASSES]), fill_value=0)

    gap=0.02
    usable=1.0-gap*(len(CLASSES)-1)

    def positions(series):
        heights={c:(series[c]/total)*usable for c in CLASSES}
        y0={}
        y=0.0
        for c in CLASSES:
            y0[c]=y
            y += heights[c] + gap
        return y0, heights

    y1,h1=positions(s1); y2,h2=positions(s2); y3,h3=positions(s3)

    x1=0.05; x2=0.47; x3=0.89; w=0.06
    cycle=plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if len(cycle)==0: cycle=["C0","C1","C2","C3"]
    color_map={c:cycle[i%len(cycle)] for i,c in enumerate(CLASSES)}

    fig, ax=plt.subplots(figsize=(12,6))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title(title)

    for c in CLASSES:
        ax.add_patch(Rectangle((x1,y1[c]), w, h1[c], alpha=0.25))
        ax.add_patch(Rectangle((x2,y2[c]), w, h2[c], alpha=0.25))
        ax.add_patch(Rectangle((x3,y3[c]), w, h3[c], alpha=0.25))
        ax.text(x1-0.01, y1[c]+h1[c]/2, c, ha="right", va="center", fontsize=10)

    ax.text(x1+w/2, 1.02, "Ext1", ha="center", va="bottom", fontsize=11, transform=ax.transAxes)
    ax.text(x2+w/2, 1.02, "Ext2", ha="center", va="bottom", fontsize=11, transform=ax.transAxes)
    ax.text(x3+w/2, 1.02, "Retrieval", ha="center", va="bottom", fontsize=11, transform=ax.transAxes)

    def add_flow(xa, ya0, ya1, xb, yb0, yb1, color):
        x0=xa+w; x1_=xb; ctrl=0.12
        verts=[
            (x0, ya0),
            (x0+ctrl, ya0),
            (x1_-ctrl, yb0),
            (x1_, yb0),
            (x1_, yb1),
            (x1_-ctrl, yb1),
            (x0+ctrl, ya1),
            (x0, ya1),
            (x0, ya0),
        ]
        codes=[MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
               MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
               MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts,codes), facecolor=color, alpha=0.35, edgecolor="none"))

    src_off={c:0.0 for c in CLASSES}
    tgt_off={c:0.0 for c in CLASSES}
    for s in CLASSES:
        for t in CLASSES:
            v=int(l12[(s,t)])
            if v<=0: continue
            h=(v/total)*usable
            ya0=y1[s]+src_off[s]; ya1=ya0+h; src_off[s]+=h
            yb0=y2[t]+tgt_off[t]; yb1=yb0+h; tgt_off[t]+=h
            add_flow(x1, ya0, ya1, x2, yb0, yb1, color_map[s])

    src_off2={c:0.0 for c in CLASSES}
    tgt_off2={c:0.0 for c in CLASSES}
    for s in CLASSES:
        for t in CLASSES:
            v=int(l2r[(s,t)])
            if v<=0: continue
            h=(v/total)*usable
            ya0=y2[s]+src_off2[s]; ya1=ya0+h; src_off2[s]+=h
            yb0=y3[t]+tgt_off2[t]; yb1=yb0+h; tgt_off2[t]+=h
            add_flow(x2, ya0, ya1, x3, yb0, yb1, color_map[s])

    out_svg=os.path.join(out_dir, f"{out_prefix}.svg")
    out_pdf=os.path.join(out_dir, f"{out_prefix}.pdf")
    out_png=os.path.join(out_dir, f"{out_prefix}.png")
    plt.tight_layout()
    plt.savefig(out_svg)
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_svg, out_pdf, out_png

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="pipeline_config.json")
    args=ap.parse_args()
    cfg=load_config(args.config)

    data_dir=cfg["paths"]["data_dir"]
    female_set=set(cfg["cohort_sex_map"]["female"])
    male_set=set(cfg["cohort_sex_map"]["male"])

    ext1_zip=os.path.join(data_dir, cfg["paths"]["ext1_classes_zip"])
    ext2_zip=os.path.join(data_dir, cfg["paths"]["ext2_classes_zip"])
    ret_zip =os.path.join(data_dir, cfg["paths"]["ret_classes_zip"])

    tmp=os.path.join(data_dir,"_tmp_ext1_ext2_ret")
    ext1_dir=os.path.join(tmp,"ext1"); ext2_dir=os.path.join(tmp,"ext2"); ret_dir=os.path.join(tmp,"ret")
    for d in [ext1_dir, ext2_dir, ret_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ext1_zip,"r") as zf: zf.extractall(ext1_dir)
    with zipfile.ZipFile(ext2_zip,"r") as zf: zf.extractall(ext2_dir)
    with zipfile.ZipFile(ret_zip ,"r") as zf: zf.extractall(ret_dir)

    ext1_cls=load_classes_from_extracted(ext1_dir,"*_ext1_onset_evoked_cell_classes.csv")
    ext2_cls=load_classes_from_extracted(ext2_dir,"*_ext2_onset_evoked_cell_classes.csv")
    ret_cls =load_classes_from_extracted(ret_dir ,"*_ret_onset_evoked_cell_classes.csv")

    animals=sorted(set(ext1_cls).intersection(ext2_cls).intersection(ret_cls), key=lambda x:int(x))

    merged=[]
    summ=[]
    for a in animals:
        d1=ext1_cls[a].rename(columns={"class":"ext1_class"})
        d2=ext2_cls[a].rename(columns={"class":"ext2_class"})
        dr=ret_cls[a].rename(columns={"class":"ret_class"})
        m=d1.merge(d2, on="cell_id", how="inner").merge(dr, on="cell_id", how="inner")
        if len(m)==0:
            continue
        m["animal_id"]=a
        m["sex"]=sex_of(a, female_set, male_set)
        merged.append(m)
        summ.append({
            "animal_id":a,
            "sex":sex_of(a, female_set, male_set),
            "n_cells_ext1":len(d1),
            "n_cells_ext2":len(d2),
            "n_cells_ret":len(dr),
            "n_cells_in_all3":len(m),
            "pct_ext1_kept_for_all3": (len(m)/len(d1))*100 if len(d1)>0 else np.nan
        })

    trans_long=pd.concat(merged, ignore_index=True) if merged else pd.DataFrame(columns=["cell_id","ext1_class","ext2_class","ret_class","animal_id","sex"])
    animal_summary=pd.DataFrame(summ).sort_values("animal_id")

    out_long=os.path.join(data_dir,"Ext1_Ext2_Ret_transitions_cellLevel_long.csv")
    out_animal=os.path.join(data_dir,"Ext1_Ext2_Ret_overlap_animalSummary.csv")
    trans_long.to_csv(out_long, index=False)
    animal_summary.to_csv(out_animal, index=False)

    pair12=trans_long.groupby(["ext1_class","ext2_class"]).size().reset_index(name="n_cells")
    pair2r=trans_long.groupby(["ext2_class","ret_class"]).size().reset_index(name="n_cells")
    triple=trans_long.groupby(["ext1_class","ext2_class","ret_class"]).size().reset_index(name="n_cells")

    pair12.to_csv(os.path.join(data_dir,"Ext1toExt2_transition_counts_cellWeighted_all3overlap.csv"), index=False)
    pair2r.to_csv(os.path.join(data_dir,"Ext2toRet_transition_counts_cellWeighted_all3overlap.csv"), index=False)
    triple.to_csv(os.path.join(data_dir,"Ext1_Ext2_Ret_transition_counts_triple_cellWeighted.csv"), index=False)

    make_plotly_sankey(trans_long, "Ext1 → Ext2 → Retrieval transitions (cells present in all 3; pooled cell-weighted)",
                       os.path.join(data_dir,"Ext1_Ext2_Ret_alluvial_cellWeighted_ALL.html"))
    make_plotly_sankey(trans_long[trans_long["sex"]=="Female"], "Ext1 → Ext2 → Retrieval transitions (Female; pooled cell-weighted)",
                       os.path.join(data_dir,"Ext1_Ext2_Ret_alluvial_cellWeighted_Female.html"))
    make_plotly_sankey(trans_long[trans_long["sex"]=="Male"], "Ext1 → Ext2 → Retrieval transitions (Male; pooled cell-weighted)",
                       os.path.join(data_dir,"Ext1_Ext2_Ret_alluvial_cellWeighted_Male.html"))

    alluvial_three_stage(trans_long, "Ext1 → Ext2 → Retrieval (cells present in all 3; pooled cell-weighted)",
                         "Ext1_Ext2_Ret_alluvial_cellWeighted_ALL", data_dir)
    alluvial_three_stage(trans_long[trans_long["sex"]=="Female"], "Ext1 → Ext2 → Retrieval (Female; pooled cell-weighted)",
                         "Ext1_Ext2_Ret_alluvial_cellWeighted_Female", data_dir)
    alluvial_three_stage(trans_long[trans_long["sex"]=="Male"], "Ext1 → Ext2 → Retrieval (Male; pooled cell-weighted)",
                         "Ext1_Ext2_Ret_alluvial_cellWeighted_Male", data_dir)

    print("Wrote transition outputs to:", data_dir)

if __name__=="__main__":
    main()
