#!/usr/bin/env python3
"""
Generate Ext2 -> Retrieval source-class transition plots.

Creates alluvial and sankey plots for Ext2 source classes:
- EarlyOnly
- LateOnly

For each source class, outputs are split by:
- All
- Female
- Male

Inputs:
- Ext2_cellClassifications_long.csv
- Ret_cellClassifications_long.csv
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


VALID = ["EarlyOnly", "Overlap", "LateOnly", "Neither"]
SOURCE_CLASSES = ["EarlyOnly", "LateOnly"]

COLOR = {
    "EarlyOnly": (1.0, 0.0, 0.0),
    "Overlap": (0.5, 0.5, 0.5),
    "LateOnly": (0.0, 0.0, 1.0),
    "Neither": (0.75, 0.75, 0.75),
}

FEMALE = {"4", "6", "8", "10", "10488", "10489", "10490", "10491", "11794"}
MALE = {"1", "2", "3", "5", "9", "10492", "10481", "10482", "11799"}


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def norm_animal_id(a: str) -> str:
    s = str(a).strip()
    if s.lower().startswith("animal"):
        s = s[6:]
    if s.lower() == "cell":
        return "10492"
    return s


def sex_of(animal_id: str) -> str:
    a = norm_animal_id(animal_id)
    if a in FEMALE:
        return "Female"
    if a in MALE:
        return "Male"
    return "Unknown"


def normalize_cell_id_any(cid: str) -> str | None:
    s = str(cid).strip()
    if not s:
        return None
    m = re.findall(r"(\d+)", s)
    if m:
        return str(int(m[-1]))
    return None


def normalize_class(x: str) -> str:
    s = str(x).strip()
    key = s.replace(" ", "").lower()
    mapping = {
        "early_only": "EarlyOnly",
        "earlyonly": "EarlyOnly",
        "late_only": "LateOnly",
        "lateonly": "LateOnly",
        "overlap": "Overlap",
        "neither": "Neither",
    }
    return mapping.get(key, s)


def _stack_positions(order, totals):
    y0 = {}
    cur = 0.0
    for k in order:
        h = float(totals.get(k, 0.0))
        y0[k] = cur
        cur += h
    return y0, cur


def alluvial_plot(link_df: pd.DataFrame, value_col: str, title: str, out_base: str):
    df = link_df.copy()
    df = df[df[value_col] > 0].copy()
    if df.empty:
        return

    total = float(df[value_col].sum())
    df["_h"] = df[value_col] / total

    left_order = sorted(df["src"].unique().tolist())
    right_order = VALID

    left_tot = df.groupby("src")["_h"].sum().reindex(left_order).fillna(0.0).to_dict()
    right_tot = df.groupby("tgt")["_h"].sum().reindex(right_order).fillna(0.0).to_dict()
    left_y0, H = _stack_positions(left_order, left_tot)
    right_y0, _ = _stack_positions(right_order, right_tot)

    loff = {k: 0.0 for k in left_order}
    roff = {k: 0.0 for k in right_order}

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    xL0, xL1 = 0.12, 0.18
    xR0, xR1 = 0.82, 0.88
    x0, x1 = xL1, xR0
    c1, c2 = 0.45, 0.55

    df = df.sort_values(["src", "tgt"]).reset_index(drop=True)
    for _, r in df.iterrows():
        src, tgt, h = r["src"], r["tgt"], float(r["_h"])
        y0L = left_y0[src] + loff[src]
        y1L = y0L + h
        y0R = right_y0[tgt] + roff[tgt]
        y1R = y0R + h
        loff[src] += h
        roff[tgt] += h

        verts = [
            (x0, y0L),
            (c1, y0L),
            (c2, y0R),
            (x1, y0R),
            (x1, y1R),
            (c2, y1R),
            (c1, y1L),
            (x0, y1L),
            (x0, y0L),
        ]
        codes = [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        patch = PathPatch(
            MplPath(verts, codes),
            facecolor=COLOR.get(src, (0.2, 0.2, 0.2)),
            alpha=0.35,
            edgecolor="black",
            linewidth=0.25,
        )
        ax.add_patch(patch)

    for k in left_order:
        h = left_tot.get(k, 0.0)
        ax.add_patch(
            Rectangle(
                (xL0, left_y0[k]),
                xL1 - xL0,
                h,
                facecolor=COLOR.get(k, (0.5, 0.5, 0.5)),
                alpha=0.85,
                edgecolor="black",
                linewidth=0.6,
            )
        )
        ax.text(
            xL0 - 0.02,
            left_y0[k] + h / 2,
            f"Ext2 {k} ({100.0*h:.1f}%)",
            ha="right",
            va="center",
            fontsize=10,
        )

    for k in right_order:
        h = right_tot.get(k, 0.0)
        ax.add_patch(
            Rectangle(
                (xR0, right_y0[k]),
                xR1 - xR0,
                h,
                facecolor=COLOR.get(k, (0.5, 0.5, 0.5)),
                alpha=0.85,
                edgecolor="black",
                linewidth=0.6,
            )
        )
        ax.text(
            xR1 + 0.02,
            right_y0[k] + h / 2,
            f"Ret {k} ({100.0*h:.1f}%)",
            ha="left",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_base + ".pdf")
    fig.savefig(out_base + ".svg")
    fig.savefig(out_base + ".png", dpi=300)
    plt.close(fig)


def sankey_plot(link_df: pd.DataFrame, title: str, out_html: str, value_col: str = "count"):
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    df = link_df.copy()
    df = df[df[value_col] > 0].copy()
    if df.empty:
        return False

    src_name = sorted(df["src"].unique().tolist())[0]
    total = float(df[value_col].sum())
    pct_by_tgt = (100.0 * df.groupby("tgt")[value_col].sum() / max(total, 1e-12)).reindex(VALID, fill_value=0.0)
    left = [f"Ext2 {src_name}"]
    right = [f"Ret {k} ({pct_by_tgt[k]:.1f}%)" for k in VALID]
    labels = left + right
    idx = {k: i for i, k in enumerate(labels)}

    sources = []
    targets = []
    values = []
    for _, r in df.iterrows():
        tgt = f"Ret {r['tgt']} ({pct_by_tgt[r['tgt']]:.1f}%)"
        if tgt not in idx:
            continue
        sources.append(idx[f"Ext2 {src_name}"])
        targets.append(idx[tgt])
        values.append(float(r[value_col]))

    colors = [f"rgba({int(255*COLOR[src_name][0])},{int(255*COLOR[src_name][1])},{int(255*COLOR[src_name][2])},0.45)"] * len(values)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=labels, pad=15, thickness=14, line=dict(color="black", width=0.6)),
                link=dict(source=sources, target=targets, value=values, color=colors),
            )
        ]
    )
    fig.update_layout(title_text=title, font_size=12)
    fig.write_html(out_html, include_plotlyjs="cdn")
    return True


def load_class_long(path: str, cls_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    need = {"animal_id", "cell_id", "class"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {sorted(need)}")
    df = df[["animal_id", "cell_id", "class"]].copy()
    df["animal_id"] = df["animal_id"].astype(str).map(norm_animal_id)
    df["cell_id"] = df["cell_id"].astype(str).map(normalize_cell_id_any)
    df["class"] = df["class"].astype(str).map(normalize_class)
    df = df[df["class"].isin(VALID)].copy()
    df = df.rename(columns={"class": cls_col_name})
    return df


def animal_weighted_mean_pct(d: pd.DataFrame, src_class: str) -> pd.DataFrame:
    """
    Compute animal-weighted mean percentages of Ret classes, conditional on source class.
    """
    if d.empty:
        return pd.DataFrame(columns=["src", "tgt", "pct", "n_animals"])
    per_animal = d.groupby(["animal_id", "ret_class"]).size().unstack(fill_value=0).reindex(columns=VALID, fill_value=0)
    per_animal_prop = per_animal.div(per_animal.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    mean_pct = 100.0 * per_animal_prop.mean(axis=0)
    out = pd.DataFrame({
        "src": src_class,
        "tgt": VALID,
        "pct": [float(mean_pct[k]) for k in VALID],
        "n_animals": int(per_animal_prop.shape[0]),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext2_long", default="Ext2_cellClassifications_long.csv")
    ap.add_argument("--ret_long", default="Ret_cellClassifications_long.csv")
    ap.add_argument("--out_dir", default="ext2_ret_transition_output")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    ext2 = load_class_long(args.ext2_long, "ext2_class")
    ret = load_class_long(args.ret_long, "ret_class")

    merged = ext2.merge(ret, on=["animal_id", "cell_id"], how="inner")
    merged["sex"] = merged["animal_id"].map(sex_of)
    merged.to_csv(os.path.join(out_dir, "Ext2toRet_transitions_cellLevel_long_fromClassLong.csv"), index=False)

    subsets = [("All", None), ("Female", "Female"), ("Male", "Male")]
    for src_class in SOURCE_CLASSES:
        for label, sx in subsets:
            d = merged[merged["ext2_class"] == src_class].copy()
            if sx is not None:
                d = d[d["sex"] == sx].copy()

            counts = d.groupby("ret_class").size().reindex(VALID, fill_value=0).reset_index()
            counts.columns = ["tgt", "count"]
            counts["src"] = src_class
            link = counts[["src", "tgt", "count"]].copy()

            counts_out = os.path.join(out_dir, f"Ext2toRet_transition_counts_{src_class}_{label}.csv")
            link.to_csv(counts_out, index=False)

            ttl = f"Ext2 {src_class} -> Ret class ({label}; cell-weighted)"
            out_base = os.path.join(out_dir, f"Ext2toRet_alluvial_{src_class}_{label}")
            alluvial_plot(link, "count", ttl, out_base)

            out_html = os.path.join(out_dir, f"Ext2toRet_Sankey_{src_class}_{label}.html")
            sankey_plot(link, ttl, out_html, value_col="count")

            # Animal-weighted mean percentage version
            aw = animal_weighted_mean_pct(d, src_class)
            aw_out = os.path.join(out_dir, f"Ext2toRet_transition_props_animalWeighted_{src_class}_{label}.csv")
            aw.to_csv(aw_out, index=False)

            ttl_aw = f"Ext2 {src_class} -> Ret class ({label}; animal-weighted mean %)"
            out_base_aw = os.path.join(out_dir, f"Ext2toRet_alluvial_{src_class}_{label}_animalWeighted_meanPct")
            alluvial_plot(aw[["src", "tgt", "pct"]], "pct", ttl_aw, out_base_aw)

            out_html_aw = os.path.join(out_dir, f"Ext2toRet_Sankey_{src_class}_{label}_animalWeighted_meanPct.html")
            sankey_plot(aw[["src", "tgt", "pct"]], ttl_aw, out_html_aw, value_col="pct")

    print(f"Wrote Ext2->Ret source-class plots to: {out_dir}")


if __name__ == "__main__":
    main()
