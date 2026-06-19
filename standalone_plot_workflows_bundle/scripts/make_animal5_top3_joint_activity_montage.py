#!/usr/bin/env python3
"""Create animal5 montage for the highest trio present in all three sessions.

The trio is chosen manually from the top of the FOV among cells present in all sessions:
C001, C015, C087.

For each session, select the frame where all three cells are simultaneously active using
z >= 0.6, maximizing the sum of their z scores among qualifying frames.
Only these three outlines are drawn.
"""

import os
from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRIN_DATA_DIR", str(REPO_ROOT)))
BASE = Path(os.environ.get("GRIN_ANIMAL5_BASE_DIR", str(DATA_DIR / "animal5_assets")))

OUT_DIR = DATA_DIR / "representative_cell_images_animal5" / "top3_joint_activity_montage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CELLS = ["C001", "C015", "C087"]
THRESHOLD = 0.6
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
OUTLINE_TARGET_SIZE = 19


def class_color(label: str | None):
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    return (0, 200, 200)


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
        indices = [header.index(c) for c in cells]
        rows = []
        for row in reader:
            vals = []
            for idx in indices:
                try:
                    vals.append(float(row[idx]))
                except Exception:
                    vals.append(np.nan)
            rows.append(vals)
    return np.array(rows, dtype=float)


def pick_joint_frame(trace_matrix: np.ndarray, threshold: float):
    counts = np.nansum(trace_matrix >= threshold, axis=1)
    exact = np.where(counts == trace_matrix.shape[1])[0]
    if len(exact) == 0:
        best = int(np.nanargmax(counts))
        return best, int(counts[best]), False
    sums = np.nansum(trace_matrix[exact], axis=1)
    best = int(exact[np.nanargmax(sums)])
    return best, int(counts[best]), True


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


def draw_frame(movie_path: Path, frame_idx: int, outlines, off_x: int, off_y: int, class_map, session_label, values, exact_match):
    im = Image.open(movie_path)
    im.seek(frame_idx)
    rgb = Image.fromarray(style_frame(np.array(im))).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    for cid, (xs, ys) in outlines.items():
        color = class_color(class_map.get(cid))
        for x, y in zip(xs, ys):
            X = int(round(x + off_x))
            Y = int(round(y + off_y))
            draw.point((X, Y), fill=color)
            draw.point((X + 1, Y), fill=color)
            draw.point((X - 1, Y), fill=color)
            draw.point((X, Y + 1), fill=color)
            draw.point((X, Y - 1), fill=color)

    draw.rectangle((6, 6, 235, 40), fill=(0, 0, 0))
    tag = "all 3" if exact_match else "best available"
    draw.text((10, 10), f"{session_label}  {tag}", fill=(255, 255, 255))
    draw.text((10, 24), f"C001={values[0]:.2f}  C015={values[1]:.2f}  C087={values[2]:.2f}", fill=(255, 255, 255))
    return rgb


def main():
    centroids = read_centroids()
    (cell_w, cell_h), outlines = load_scaled_outlines(TARGET_CELLS, centroids)
    class_maps = {sess: read_class_map(sess) for sess in SESSIONS}
    offsets = {sess: compute_center_offset(MOVIES[sess], cell_w, cell_h) for sess in SESSIONS}

    panels = []
    summary_rows = []

    for sess in SESSIONS:
        trace_matrix = load_trace_matrix(TRACES[sess], TARGET_CELLS)
        csv_idx, active_count, exact_match = pick_joint_frame(trace_matrix, THRESHOLD)
        values = trace_matrix[csv_idx].tolist()

        im = Image.open(MOVIES[sess])
        n_frames = getattr(im, "n_frames", 1)
        frame_offset = n_frames - trace_matrix.shape[0]
        movie_idx = int(csv_idx + frame_offset)

        off_x, off_y = offsets[sess]
        panel = draw_frame(
            MOVIES[sess],
            movie_idx,
            outlines,
            off_x,
            off_y,
            class_maps[sess],
            sess,
            values,
            exact_match,
        )

        out_path = OUT_DIR / f"{sess}_top3_joint_frame{movie_idx:05d}.png"
        panel.save(out_path)
        panels.append(panel)
        summary_rows.append({
            "session": sess,
            "csv_frame_index": int(csv_idx),
            "movie_frame_index": movie_idx,
            "active_count_at_threshold": int(active_count),
            "exact_all_three_match": int(exact_match),
            "threshold": THRESHOLD,
            "C001_z": float(values[0]),
            "C015_z": float(values[1]),
            "C087_z": float(values[2]),
            "png": str(out_path),
        })

    if panels:
        w, h = panels[0].size
        montage = Image.new("RGB", (w * len(panels), h), (0, 0, 0))
        for i, panel in enumerate(panels):
            montage.paste(panel, (i * w, 0))
        montage_path = OUT_DIR / "animal5_top3_joint_activity_montage.png"
        montage.save(montage_path)

    summary_path = OUT_DIR / "animal5_top3_joint_activity_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session",
                "csv_frame_index",
                "movie_frame_index",
                "active_count_at_threshold",
                "exact_all_three_match",
                "threshold",
                "C001_z",
                "C015_z",
                "C087_z",
                "png",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
