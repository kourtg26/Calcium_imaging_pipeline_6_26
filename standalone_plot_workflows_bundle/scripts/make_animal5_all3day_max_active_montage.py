#!/usr/bin/env python3
"""Create animal5 montage using cells present on all three days.

For each session, choose the frame with the most active cells among the set of
registered cells that are present in Ext1, Ext2, and Ret. Activity is defined
as z >= 0.2. All all-3-day cells are outlined and active cells are drawn thicker.
"""

import os
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRIN_DATA_DIR", str(REPO_ROOT)))
BASE = Path(os.environ.get("GRIN_ANIMAL5_BASE_DIR", str(DATA_DIR / "animal5_assets")))
OUT_DIR = DATA_DIR / "representative_cell_images_animal5" / "all3day_max_active_montage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = ["Ext1", "Ext2", "Ret"]
THRESHOLD = 0.2

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
OUTLINE_TARGET_SIZE = 19


def class_color(label: str | None):
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    return (0, 200, 200)


def read_all3day_cells():
    props = BASE / "cell_traces_registered_cells_all_days_animal5-props.csv"
    cells = []
    with props.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ActiveSegment0") == "1" and row.get("ActiveSegment1") == "1" and row.get("ActiveSegment2") == "1":
                cells.append(row["Name"])
    return cells


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
    props = BASE / "cell_traces_registered_cells_all_days_animal5-props.csv"
    out = {}
    with props.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name")
            if not name:
                continue
            try:
                out[name] = (float(row["CentroidX"]), float(row["CentroidY"]))
            except Exception:
                continue
    return out


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
        raise RuntimeError("No cell images found")

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
    return available, np.array(rows, dtype=float)


def choose_best_frame(trace_matrix: np.ndarray):
    active_counts = np.nansum(trace_matrix >= THRESHOLD, axis=1)
    positive_sums = np.nansum(np.where(trace_matrix > 0, trace_matrix, 0), axis=1)
    best_count = int(np.nanmax(active_counts))
    candidates = np.where(active_counts == best_count)[0]
    best_idx = int(candidates[np.nanargmax(positive_sums[candidates])])
    active_mask = trace_matrix[best_idx] >= THRESHOLD
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


def draw_panel(movie_path: Path, movie_frame_idx: int, outlines, off_x: int, off_y: int, class_map, active_cells, session_label, active_count):
    im = Image.open(movie_path)
    im.seek(movie_frame_idx)
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

    draw.rectangle((6, 6, 175, 22), fill=(0, 0, 0))
    draw.text((10, 10), f"{session_label}  active={active_count}", fill=(255, 255, 255))
    return rgb


def main():
    all3_cells = read_all3day_cells()
    centroids = read_centroids()
    (cell_w, cell_h), outlines = load_scaled_outlines(all3_cells, centroids)
    class_maps = {sess: read_class_map(sess) for sess in SESSIONS}
    offsets = {sess: compute_center_offset(MOVIES[sess], cell_w, cell_h) for sess in SESSIONS}

    panels = []
    summary_rows = []

    for sess in SESSIONS:
        available_cells, trace_matrix = load_trace_matrix(TRACES[sess], all3_cells)
        csv_idx, active_mask, active_count = choose_best_frame(trace_matrix)
        active_cells = {available_cells[i] for i, flag in enumerate(active_mask) if flag}

        im = Image.open(MOVIES[sess])
        n_frames = getattr(im, "n_frames", 1)
        frame_offset = n_frames - trace_matrix.shape[0]
        movie_idx = int(csv_idx + frame_offset)
        off_x, off_y = offsets[sess]

        panel = draw_panel(
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

        out_path = OUT_DIR / f"{sess}_all3day_max_active_frame{movie_idx:05d}.png"
        panel.save(out_path)
        panels.append(panel)
        summary_rows.append({
            "session": sess,
            "csv_frame_index": int(csv_idx),
            "movie_frame_index": movie_idx,
            "active_count_at_threshold": int(active_count),
            "threshold": THRESHOLD,
            "active_cells": ";".join(sorted(active_cells)),
            "png": str(out_path),
        })

    if panels:
        w, h = panels[0].size
        montage = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
        for i, panel in enumerate(panels):
            montage.paste(panel, (i * w, 0))
        montage.save(OUT_DIR / "animal5_all3day_max_active_montage.png")

    with (OUT_DIR / "animal5_all3day_max_active_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session", "csv_frame_index", "movie_frame_index", "active_count_at_threshold", "threshold", "active_cells", "png"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
