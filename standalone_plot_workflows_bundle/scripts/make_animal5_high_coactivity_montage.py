#!/usr/bin/env python3
"""Create animal5 Ext1/Ext2/Ret montage using the frame with the most strongly active
union representative cells in each session.

Strongly active is defined as z >= 2 in the session z-scored trace file.
All union representative cells are outlined. Cells active in the chosen frame are
drawn thicker.
"""

import os
from pathlib import Path
import csv
import math
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRIN_DATA_DIR", str(REPO_ROOT)))
BASE = Path(os.environ.get("GRIN_ANIMAL5_BASE_DIR", str(DATA_DIR / "animal5_assets")))

SUMMARY_PATH = DATA_DIR / "representative_traces_top5_peak_animal1_animal5_individual/selected_top5_peak_cells_animal1_animal5.csv"
OUT_DIR = DATA_DIR / "representative_cell_images_animal5" / "high_coactivity_montage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = ["Ext1", "Ext2", "Ret"]

MOVIES = {
    "Ext1": BASE / "ext1_neural_activity_frames.tiff",
    "Ext2": BASE / "ext2_neural_activity_frames.tiff",
    "Ret": BASE / "ret_neural_activity_frames.tiff",
}

TRACES = {
    "Ext1": BASE / "animal5_extinction1_zscored_presentcells.csv",
    "Ext2": BASE / "animal5_extinction2_zscored_presentcells.csv",
    "Ret": BASE / "animal5_retrieval_zscored_presentcells.csv",
}

CLASS_FILES = {
    "Ext1": DATA_DIR / "Ext1_cellClassifications_long.csv",
    "Ext2": DATA_DIR / "Ext2_cellClassifications_long.csv",
    "Ret": DATA_DIR / "Ret_cellClassifications_long.csv",
}

CLASS_COLORS = {
    "EarlyOnly": (255, 0, 0),
    "LateOnly": (0, 102, 255),
}

BG_SCALE = 0.8
CONTRAST_FACTOR = 1.4
ACTIVE_Z_THRESHOLD = 2.0
OUTLINE_TARGET_SIZE = 19


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


def load_scaled_outlines(cells, centroids):
    outlines = {}
    cell_size = None
    for cid in cells:
        fp = BASE / f"cell_images_registered_cells_animal5_{cid}.tiff"
        if not fp.exists():
            continue
        img = np.array(Image.open(fp))
        if img.ndim > 2:
            img = img[..., 0]
        if cell_size is None:
            cell_size = (img.shape[1], img.shape[0])
        mask = img > 0
        if not np.any(mask):
            continue
        up = np.pad(mask, ((1, 0), (0, 0)), mode="constant")[:-1, :]
        down = np.pad(mask, ((0, 1), (0, 0)), mode="constant")[1:, :]
        left = np.pad(mask, ((0, 0), (1, 0)), mode="constant")[:, :-1]
        right = np.pad(mask, ((0, 0), (0, 1)), mode="constant")[:, 1:]
        edge = mask & ~(up & down & left & right)
        ys, xs = np.where(edge)
        outlines[cid] = (xs, ys)

    if cell_size is None:
        raise RuntimeError("No cell images found for outlines")

    scaled = {}
    for cid, (xs, ys) in outlines.items():
        centroid = centroids.get(cid)
        if centroid is None:
            continue
        cx, cy = centroid
        dx = xs.astype(float) - cx
        dy = ys.astype(float) - cy
        if dx.size == 0:
            continue
        orig_w = float(dx.max() - dx.min() + 1.0)
        orig_h = float(dy.max() - dy.min() + 1.0)
        if orig_w <= 0 or orig_h <= 0:
            scale = 1.0
        else:
            scale = min((OUTLINE_TARGET_SIZE - 1) / orig_w, (OUTLINE_TARGET_SIZE - 1) / orig_h)
        scaled[cid] = (cx + dx * scale, cy + dy * scale)
    return cell_size, scaled


def compute_center_offset(movie_path: Path, cell_w: int, cell_h: int):
    frame0 = np.array(Image.open(movie_path))
    h, w = frame0.shape[:2]
    return int((w - cell_w) // 2), int((h - cell_h) // 2)


def load_trace_matrix(trace_path: Path, cells):
    with trace_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        available = [c for c in cells if c in header]
        indices = [header.index(c) for c in available]
        rows = []
        for row in reader:
            vals = []
            for idx in indices:
                try:
                    vals.append(float(row[idx]))
                except Exception:
                    vals.append(np.nan)
            rows.append(vals)
    X = np.array(rows, dtype=float) if rows else np.empty((0, 0), dtype=float)
    return available, X


def pick_best_frame(trace_matrix: np.ndarray):
    if trace_matrix.size == 0:
        return 0, np.array([], dtype=bool), 0
    active_counts = np.nansum(trace_matrix >= ACTIVE_Z_THRESHOLD, axis=1)
    positive_sums = np.nansum(np.where(trace_matrix > 0, trace_matrix, 0), axis=1)
    best_count = int(np.nanmax(active_counts))
    candidates = np.where(active_counts == best_count)[0]
    if len(candidates) == 0:
        return 0, np.zeros(trace_matrix.shape[1], dtype=bool), 0
    best_idx = int(candidates[np.nanargmax(positive_sums[candidates])])
    active_mask = trace_matrix[best_idx] >= ACTIVE_Z_THRESHOLD
    return best_idx, active_mask, best_count


def style_frame(frame: np.ndarray):
    arr = frame.astype(float)
    p1, p99 = np.nanpercentile(arr, [1, 99])
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
        p1, p99 = 0.0, 1.0
    arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
    arr = np.clip(arr * BG_SCALE, 0, 1)
    arr = np.clip((arr - 0.5) * CONTRAST_FACTOR + 0.5, 0, 1)
    return (arr * 255).astype(np.uint8)


def draw_frame(movie_path: Path, frame_idx: int, outlines, off_x: int, off_y: int, class_map, active_cells, session_label, count_label):
    im = Image.open(movie_path)
    im.seek(frame_idx)
    rgb = Image.fromarray(style_frame(np.array(im))).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    for cid, (xs, ys) in outlines.items():
        color = class_color(class_map.get(cid))
        width = 2 if cid in active_cells else 1
        for x, y in zip(xs, ys):
            X = int(round(x + off_x))
            Y = int(round(y + off_y))
            if width == 1:
                draw.point((X, Y), fill=color)
            else:
                draw.point((X, Y), fill=color)
                draw.point((X + 1, Y), fill=color)
                draw.point((X - 1, Y), fill=color)
                draw.point((X, Y + 1), fill=color)
                draw.point((X, Y - 1), fill=color)

    draw.rectangle((6, 6, 168, 34), fill=(0, 0, 0))
    draw.text((10, 10), f"{session_label}  active={count_label}", fill=(255, 255, 255))
    return rgb


def main():
    union_cells = read_union_rep_cells(SUMMARY_PATH)
    centroids = read_centroids()
    (cell_w, cell_h), outlines = load_scaled_outlines(union_cells, centroids)
    class_maps = {sess: read_class_map(sess) for sess in SESSIONS}
    offsets = {sess: compute_center_offset(MOVIES[sess], cell_w, cell_h) for sess in SESSIONS}

    rendered = []
    summary_rows = []

    for sess in SESSIONS:
        available_cells, trace_matrix = load_trace_matrix(TRACES[sess], union_cells)
        best_csv_idx, active_mask, active_count = pick_best_frame(trace_matrix)
        active_cells = {available_cells[i] for i, flag in enumerate(active_mask) if flag}

        im = Image.open(MOVIES[sess])
        n_frames = getattr(im, "n_frames", 1)
        frame_offset = n_frames - trace_matrix.shape[0]
        movie_idx = int(best_csv_idx + frame_offset)
        off_x, off_y = offsets[sess]

        img = draw_frame(
            MOVIES[sess],
            movie_idx,
            outlines,
            off_x,
            off_y,
            class_maps[sess],
            active_cells,
            sess,
            active_count,
        )

        out_path = OUT_DIR / f"{sess}_high_coactivity_frame{movie_idx:05d}.png"
        img.save(out_path)
        rendered.append((sess, img))
        summary_rows.append({
            "session": sess,
            "csv_frame_index": int(best_csv_idx),
            "movie_frame_index": movie_idx,
            "active_cell_count_z_ge_2": int(active_count),
            "active_cells": ";".join(sorted(active_cells)),
            "png": str(out_path),
        })

    if rendered:
        panel_w, panel_h = rendered[0][1].size
        montage = Image.new("RGB", (panel_w * len(rendered), panel_h), (0, 0, 0))
        for i, (_, img) in enumerate(rendered):
            montage.paste(img, (i * panel_w, 0))
        montage_path = OUT_DIR / "animal5_Ext1_Ext2_Ret_high_coactivity_montage.png"
        montage.save(montage_path)

    summary_path = OUT_DIR / "animal5_high_coactivity_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session", "csv_frame_index", "movie_frame_index", "active_cell_count_z_ge_2", "active_cells", "png"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
