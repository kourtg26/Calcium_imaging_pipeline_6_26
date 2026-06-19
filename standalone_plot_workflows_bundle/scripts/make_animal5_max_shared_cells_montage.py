#!/usr/bin/env python3
"""Create animal5 montage using the largest cell subset shared across sessions at z >= 0.6.

The shared subset is defined over the union representative cells. For each session,
the script chooses the frame where that entire subset is active and the subset's summed
z score is maximal.
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
OUT_DIR = DATA_DIR / "representative_cell_images_animal5" / "max_shared_cells_montage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = ["Ext1", "Ext2", "Ret"]
THRESHOLD = 0.6

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


def read_union_rep_cells(path: Path):
    cells = []
    seen = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("animal") == "animal5" and row.get("cell_id") and row["cell_id"] not in seen:
                seen.add(row["cell_id"])
                cells.append(row["cell_id"])
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
    path = BASE / "cell_traces_registered_cells_all_days_animal5-props.csv"
    out = {}
    with path.open("r", newline="") as f:
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


def session_masks(trace_matrix: np.ndarray):
    masks = []
    for row in trace_matrix:
        mask = 0
        for bit, value in enumerate(row):
            if np.isfinite(value) and value >= THRESHOLD:
                mask |= (1 << bit)
        masks.append(mask)
    return masks


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


def choose_best_frame(trace_matrix: np.ndarray, subset_mask: int):
    subset_bits = [i for i in range(trace_matrix.shape[1]) if subset_mask & (1 << i)]
    qualifying = []
    for frame_idx, row in enumerate(trace_matrix):
        ok = True
        for bit in subset_bits:
            value = row[bit]
            if not np.isfinite(value) or value < THRESHOLD:
                ok = False
                break
        if ok:
            qualifying.append((frame_idx, float(np.nansum(row[subset_bits]))))
    if not qualifying:
        raise RuntimeError("No qualifying frame found for subset")
    return max(qualifying, key=lambda x: x[1])


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


def label_position(x: int, y: int, index: int):
    dx = 10 + (index % 2) * 18
    dy = -18 - (index % 3) * 10
    return x + dx, y + dy


def draw_panel(movie_path: Path, movie_frame_idx: int, outlines, off_x: int, off_y: int, class_map, shared_cells, session_label):
    im = Image.open(movie_path)
    im.seek(movie_frame_idx)
    rgb = Image.fromarray(style_frame(np.array(im))).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    for cid, (xs, ys) in outlines.items():
        color = class_color(class_map.get(cid))
        width = 2 if cid in shared_cells else 1
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

    for i, cid in enumerate(shared_cells):
        cx, cy = outlines[cid]
        # use median outline point for label anchor
        x = int(round(np.median(cx) + off_x))
        y = int(round(np.median(cy) + off_y))
        tx, ty = label_position(x, y, i)
        draw.rectangle((tx - 2, ty - 2, tx + 34, ty + 10), fill=(0, 0, 0))
        draw.text((tx, ty), cid, fill=(255, 255, 255))

    draw.rectangle((6, 6, 220, 22), fill=(0, 0, 0))
    draw.text((10, 10), f"{session_label}  shared={len(shared_cells)}  z>={THRESHOLD}", fill=(255, 255, 255))
    return rgb


def main():
    union_cells = read_union_rep_cells(SUMMARY_PATH)
    centroids = read_centroids()
    (cell_w, cell_h), outlines = load_scaled_outlines(union_cells, centroids)
    class_maps = {sess: read_class_map(sess) for sess in SESSIONS}
    offsets = {sess: compute_center_offset(MOVIES[sess], cell_w, cell_h) for sess in SESSIONS}

    traces = {}
    masks = {}
    available_cells = None
    for sess in SESSIONS:
        available, trace_matrix = load_trace_matrix(TRACES[sess], union_cells)
        traces[sess] = trace_matrix
        masks[sess] = session_masks(trace_matrix)
        if available_cells is None:
            available_cells = available

    common = subset_closure(masks["Ext1"]) & subset_closure(masks["Ext2"]) & subset_closure(masks["Ret"])
    best_mask = max(common, key=int.bit_count)
    shared_cells = decode_mask(best_mask, available_cells)

    panels = []
    summary_rows = []
    for sess in SESSIONS:
        csv_frame_idx, subset_sum = choose_best_frame(traces[sess], best_mask)
        im = Image.open(MOVIES[sess])
        n_frames = getattr(im, "n_frames", 1)
        frame_offset = n_frames - traces[sess].shape[0]
        movie_frame_idx = int(csv_frame_idx + frame_offset)
        off_x, off_y = offsets[sess]
        panel = draw_panel(
            MOVIES[sess],
            movie_frame_idx,
            outlines,
            off_x,
            off_y,
            class_maps[sess],
            shared_cells,
            sess,
        )
        out_path = OUT_DIR / f"{sess}_max_shared_frame{movie_frame_idx:05d}.png"
        panel.save(out_path)
        panels.append(panel)
        summary_rows.append({
            "session": sess,
            "csv_frame_index": int(csv_frame_idx),
            "movie_frame_index": int(movie_frame_idx),
            "shared_cells": ";".join(shared_cells),
            "shared_count": len(shared_cells),
            "subset_sum_z": subset_sum,
            "png": str(out_path),
        })

    if panels:
        w, h = panels[0].size
        montage = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
        for i, panel in enumerate(panels):
            montage.paste(panel, (i * w, 0))
        montage.save(OUT_DIR / "animal5_max_shared_cells_montage.png")

    with (OUT_DIR / "animal5_max_shared_cells_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["session", "csv_frame_index", "movie_frame_index", "shared_cells", "shared_count", "subset_sum_z", "png"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Shared cells:", ",".join(shared_cells))
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
