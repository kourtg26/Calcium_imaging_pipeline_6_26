#!/usr/bin/env python3
"""Create a montage using movie-derived ROI activity instead of trace z-scores.

Cells are limited to those present across Ext1, Ext2, and Ret. For each session we
compute the mean ROI intensity for each cell across all movie frames, z-score each
cell's ROI timecourse within session, and use those movie-derived z-scores to find
the largest subset of cells that can be simultaneously active in at least one frame
of all three sessions.

The output shows only that shared active subset.
"""

import os
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRIN_DATA_DIR", str(REPO_ROOT)))
BASE = Path(os.environ.get("GRIN_ANIMAL5_BASE_DIR", str(DATA_DIR / "animal5_assets")))
OUT_DIR = DATA_DIR / "representative_cell_images_animal5" / "all3day_movieROI_shared_active"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = ["Ext1", "Ext2", "Ret"]
ROI_Z_THRESHOLD = 1.0

MOVIES = {
    "Ext1": BASE / "ext1_neural_activity_frames.tiff",
    "Ext2": BASE / "ext2_neural_activity_frames.tiff",
    "Ret": BASE / "ret_neural_activity_frames.tiff",
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


def load_masks_and_outlines(cells, centroids):
    masks = {}
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
        masks[cid] = mask

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

    return cell_size, masks, scaled


def compute_center_offset(movie_path: Path, cell_w: int, cell_h: int):
    frame0 = np.array(Image.open(movie_path))
    h, w = frame0.shape[:2]
    return int((w - cell_w) // 2), int((h - cell_h) // 2)


def compute_roi_z(movie_path: Path, cells, masks, off_x: int, off_y: int):
    im = Image.open(movie_path)
    n_frames = getattr(im, "n_frames", 1)

    coords = {}
    for cid in cells:
        mask = masks[cid]
        ys, xs = np.where(mask)
        coords[cid] = (ys + off_y, xs + off_x)

    raw = np.full((n_frames, len(cells)), np.nan, dtype=float)
    for fi in range(n_frames):
        im.seek(fi)
        frame = np.array(im).astype(float)
        for ci, cid in enumerate(cells):
            ys, xs = coords[cid]
            roi = frame[ys, xs]
            if roi.size:
                raw[fi, ci] = float(np.nanmean(roi))

    mu = np.nanmean(raw, axis=0)
    sigma = np.nanstd(raw, axis=0)
    sigma[sigma == 0] = np.nan
    z = (raw - mu) / sigma
    return z


def subset_closure(masks):
    out = set()
    for mask in set(masks):
        sub = mask
        while sub:
            out.add(sub)
            sub = (sub - 1) & mask
    return out


def decode_mask(mask: int, cells):
    return [cells[i] for i in range(len(cells)) if mask & (1 << i)]


def choose_best_frame(z: np.ndarray, subset_mask: int):
    subset_bits = [i for i in range(z.shape[1]) if subset_mask & (1 << i)]
    best_idx = None
    best_sum = None
    for fi in range(z.shape[0]):
        vals = z[fi, subset_bits]
        if np.all(np.isfinite(vals)) and np.all(vals >= ROI_Z_THRESHOLD):
            score = float(np.nansum(vals))
            if best_sum is None or score > best_sum:
                best_sum = score
                best_idx = fi
    if best_idx is None:
        raise RuntimeError("No frame found for shared ROI subset")
    return best_idx, best_sum


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


def draw_panel(movie_path: Path, movie_frame_idx: int, outlines, off_x: int, off_y: int, class_map, active_cells, session_label):
    im = Image.open(movie_path)
    im.seek(movie_frame_idx)
    rgb = Image.fromarray(style_frame(np.array(im))).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    for cid in active_cells:
        xs, ys = outlines[cid]
        color = class_color(class_map.get(cid))
        for x, y in zip(xs, ys):
            X = int(round(x + off_x))
            Y = int(round(y + off_y))
            draw.point((X, Y), fill=color)
            draw.point((X + 1, Y), fill=color)
            draw.point((X - 1, Y), fill=color)
            draw.point((X, Y + 1), fill=color)
            draw.point((X, Y - 1), fill=color)

    draw.rectangle((6, 6, 210, 22), fill=(0, 0, 0))
    draw.text((10, 10), f"{session_label}  movie ROI z>={ROI_Z_THRESHOLD}", fill=(255, 255, 255))
    return rgb


def main():
    cells = read_all3day_cells()
    centroids = read_centroids()
    (cell_w, cell_h), masks, outlines = load_masks_and_outlines(cells, centroids)
    class_maps = {sess: read_class_map(sess) for sess in SESSIONS}
    offsets = {sess: compute_center_offset(MOVIES[sess], cell_w, cell_h) for sess in SESSIONS}

    z_by_session = {}
    mask_rows = {}
    for sess in SESSIONS:
        off_x, off_y = offsets[sess]
        z = compute_roi_z(MOVIES[sess], cells, masks, off_x, off_y)
        z_by_session[sess] = z
        rows = []
        for fi in range(z.shape[0]):
            mask = 0
            for ci, value in enumerate(z[fi]):
                if np.isfinite(value) and value >= ROI_Z_THRESHOLD:
                    mask |= (1 << ci)
            rows.append(mask)
        mask_rows[sess] = rows

    common = subset_closure(mask_rows["Ext1"]) & subset_closure(mask_rows["Ext2"]) & subset_closure(mask_rows["Ret"])
    best_mask = max(common, key=int.bit_count)
    active_cells = decode_mask(best_mask, cells)

    panels = []
    summary_rows = []
    for sess in SESSIONS:
        frame_idx, subset_sum = choose_best_frame(z_by_session[sess], best_mask)
        off_x, off_y = offsets[sess]
        panel = draw_panel(
            MOVIES[sess],
            frame_idx,
            outlines,
            off_x,
            off_y,
            class_maps[sess],
            active_cells,
            sess,
        )
        out_path = OUT_DIR / f"{sess}_movieROI_shared_frame{frame_idx:05d}.png"
        panel.save(out_path)
        panels.append(panel)
        summary_rows.append({
            "session": sess,
            "movie_frame_index": int(frame_idx),
            "roi_z_threshold": ROI_Z_THRESHOLD,
            "shared_active_cells": ";".join(active_cells),
            "shared_active_count": len(active_cells),
            "subset_sum_roi_z": subset_sum,
            "png": str(out_path),
        })

    if panels:
        w, h = panels[0].size
        montage = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
        for i, panel in enumerate(panels):
            montage.paste(panel, (i * w, 0))
        montage.save(OUT_DIR / "animal5_all3day_movieROI_shared_active_montage.png")

    with (OUT_DIR / "animal5_all3day_movieROI_shared_active_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session", "movie_frame_index", "roi_z_threshold", "shared_active_cells", "shared_active_count", "subset_sum_roi_z", "png"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Shared movie-ROI-active cells:", ",".join(active_cells))
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
