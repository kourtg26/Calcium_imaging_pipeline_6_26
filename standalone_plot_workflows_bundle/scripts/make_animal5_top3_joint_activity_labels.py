#!/usr/bin/env python3
"""Add labels for non-target cells active in all selected top3-joint frames.

Uses the existing chosen frames from top3_joint_activity_montage and labels any
union representative cells that are active in all three frames at z >= 1.0.
"""

import os
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRIN_DATA_DIR", str(REPO_ROOT)))
BASE = Path(os.environ.get("GRIN_ANIMAL5_BASE_DIR", str(DATA_DIR / "animal5_assets")))

SUMMARY_PATH = DATA_DIR / "representative_traces_top5_peak_animal1_animal5_individual/selected_top5_peak_cells_animal1_animal5.csv"
TOP3_SUMMARY_PATH = DATA_DIR / "representative_cell_images_animal5/top3_joint_activity_montage/animal5_top3_joint_activity_summary.csv"
OUT_DIR = DATA_DIR / "representative_cell_images_animal5" / "top3_joint_activity_montage"

SESSIONS = ["Ext1", "Ext2", "Ret"]
TARGET_CELLS = {"C001", "C015", "C087"}
SHARED_THRESHOLD = 1.0

CLASS_FILES = {
    "Ext1": DATA_DIR / "Ext1_cellClassifications_long.csv",
    "Ext2": DATA_DIR / "Ext2_cellClassifications_long.csv",
    "Ret": DATA_DIR / "Ret_cellClassifications_long.csv",
}

TRACES = {
    "Ext1": BASE / "animal5_extinction1_zscored_presentcells.csv",
    "Ext2": BASE / "animal5_extinction2_zscored_presentcells.csv",
    "Ret": BASE / "animal5_retrieval_zscored_presentcells.csv",
}

CLASS_COLORS = {
    "EarlyOnly": (255, 0, 0),
    "LateOnly": (0, 102, 255),
}


def class_color(label: str | None):
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    return (0, 200, 200)


def read_union_rep_cells(path: Path):
    cells = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("animal") == "animal5" and row.get("cell_id"):
                cells.append(row["cell_id"])
    return sorted(set(cells))


def read_frame_summary(path: Path):
    rows = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["session"]] = {
                "csv_frame_index": int(row["csv_frame_index"]),
                "movie_frame_index": int(row["movie_frame_index"]),
                "png": Path(row["png"]),
            }
    return rows


def read_trace_row_values(trace_path: Path, cells, frame_idx: int):
    with trace_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx = {c: header.index(c) for c in cells if c in header}
        rows = list(reader)
    row = rows[frame_idx]
    values = {}
    for cid, i in idx.items():
        try:
            values[cid] = float(row[i])
        except Exception:
            values[cid] = np.nan
    return values


def read_centroids():
    path = BASE / "cell_traces_registered_cells_all_days_animal5-props.csv"
    centroids = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name")
            if not name:
                continue
            try:
                centroids[name] = (float(row["CentroidX"]), float(row["CentroidY"]))
            except Exception:
                continue
    return centroids


def read_class_map(session: str):
    path = CLASS_FILES[session]
    out = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row.get("animal_id")
            if session == "Ret":
                if aid != "5":
                    continue
            else:
                if aid != "animal5":
                    continue
            cid = row.get("cell_id")
            if cid:
                out[cid] = row.get("class")
    return out


def read_centroid_offset():
    sample = BASE / "ext1_neural_activity_frames.tiff"
    cell_img = BASE / "cell_images_registered_cells_animal5_C001.tiff"
    img = np.array(Image.open(cell_img))
    if img.ndim > 2:
        img = img[..., 0]
    cell_w, cell_h = img.shape[1], img.shape[0]
    frame0 = np.array(Image.open(sample))
    h, w = frame0.shape[:2]
    return int((w - cell_w) // 2), int((h - cell_h) // 2)


def label_position(cx: float, cy: float, index: int):
    dx = 10 + (index % 2) * 18
    dy = -18 - (index % 3) * 10
    return int(cx + dx), int(cy + dy)


def main():
    union_cells = read_union_rep_cells(SUMMARY_PATH)
    frame_summary = read_frame_summary(TOP3_SUMMARY_PATH)
    centroids = read_centroids()
    off_x, off_y = read_centroid_offset()

    values_by_session = {}
    for sess in SESSIONS:
        values_by_session[sess] = read_trace_row_values(
            TRACES[sess],
            union_cells,
            frame_summary[sess]["csv_frame_index"],
        )

    shared = set(union_cells)
    for sess in SESSIONS:
        shared &= {
            cid for cid, val in values_by_session[sess].items()
            if np.isfinite(val) and val >= SHARED_THRESHOLD
        }
    shared -= TARGET_CELLS
    shared = sorted(shared)

    class_maps = {sess: read_class_map(sess) for sess in SESSIONS}
    panels = []

    for sess in SESSIONS:
        img = Image.open(frame_summary[sess]["png"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for i, cid in enumerate(shared):
            centroid = centroids.get(cid)
            if centroid is None:
                continue
            cx, cy = centroid
            x = int(round(cx + off_x))
            y = int(round(cy + off_y))
            color = class_color(class_maps[sess].get(cid))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
            tx, ty = label_position(x, y, i)
            draw.rectangle((tx - 2, ty - 2, tx + 34, ty + 10), fill=(0, 0, 0))
            draw.text((tx, ty), cid, fill=(255, 255, 255))

        out_path = OUT_DIR / f"{sess}_top3_joint_labeled.png"
        img.save(out_path)
        panels.append(img)

    if panels:
        w, h = panels[0].size
        montage = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
        for i, panel in enumerate(panels):
            montage.paste(panel, (i * w, 0))
        montage.save(OUT_DIR / "animal5_top3_joint_activity_montage_labeled.png")

    summary_out = OUT_DIR / "animal5_top3_joint_activity_shared_labels.csv"
    with summary_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "shared_labeled_cells"])
        writer.writeheader()
        writer.writerow({
            "threshold": SHARED_THRESHOLD,
            "shared_labeled_cells": ";".join(shared),
        })

    print("Shared labeled cells:", ",".join(shared) if shared else "none")
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
