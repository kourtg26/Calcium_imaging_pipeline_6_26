#!/usr/bin/env python3
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ANIMALS = ["1", "2", "4", "6"]
CELL_IDS = [f"C{i:03d}" for i in range(1, 7)]
FPS = 10.0
N_FRAMES = 260
TONE_ONSETS = [40, 90, 140, 190]
TONE_DURATION = 20
FREEZE_BOUTS = [(96, 112), (196, 212)]


EXT1_PATTERNS = {
    "1": {"C001": {1, 2}, "C002": {3, 4}, "C003": {1, 2, 3, 4}, "C004": set(), "C005": {1, 2}, "C006": {3, 4}},
    "2": {"C001": {1, 2}, "C002": {3, 4}, "C003": {1, 2, 3, 4}, "C004": set(), "C005": {3, 4}, "C006": set()},
    "4": {"C001": {1, 2}, "C002": {3, 4}, "C003": {1, 2, 3, 4}, "C004": set(), "C005": {1, 2}, "C006": {1, 2, 3, 4}},
    "6": {"C001": {1, 2}, "C002": {3, 4}, "C003": {1, 2, 3, 4}, "C004": set(), "C005": set(), "C006": {3, 4}},
}

EXT2_PATTERNS = {
    "1": {"C001": {1, 2, 3, 4}, "C002": {3, 4}, "C003": {1, 2}, "C004": set(), "C005": {1, 2}, "C006": set()},
    "2": {"C001": {1, 2}, "C002": {3, 4}, "C003": {1, 2, 3, 4}, "C004": set(), "C005": {3, 4}, "C006": set()},
    "4": {"C001": {1, 2, 3, 4}, "C002": {3, 4}, "C003": {1, 2}, "C004": set(), "C005": {1, 2, 3, 4}, "C006": {3, 4}},
    "6": {"C001": {1, 2}, "C002": set(), "C003": {1, 2, 3, 4}, "C004": set(), "C005": {1, 2}, "C006": {3, 4}},
}

RET_PATTERNS = {
    "1": {"C001": {1}, "C002": {4}, "C003": {1, 4}, "C004": set(), "C005": {1}, "C006": {4}},
    "2": {"C001": {1}, "C002": {4}, "C003": {1, 4}, "C004": set(), "C005": {4}, "C006": set()},
    "4": {"C001": {1, 4}, "C002": {4}, "C003": {1}, "C004": set(), "C005": {1}, "C006": {1, 4}},
    "6": {"C001": {1}, "C002": set(), "C003": {1, 4}, "C004": set(), "C005": {1}, "C006": {4}},
}


def build_common_flags() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(N_FRAMES, dtype=float) / FPS
    tone_flag = np.zeros(N_FRAMES, dtype=int)
    tone_id = np.zeros(N_FRAMES, dtype=int)
    freeze = np.zeros(N_FRAMES, dtype=int)

    for tone_num, onset in enumerate(TONE_ONSETS, start=1):
        tone_flag[onset:onset + TONE_DURATION] = 1
        tone_id[onset:onset + TONE_DURATION] = tone_num

    for start, end in FREEZE_BOUTS:
        freeze[start:end] = 1

    return time, tone_flag, tone_id, freeze


def make_signal(patterns: dict[str, set[int]], animal_id: str, scale: float, z_like: bool) -> np.ndarray:
    rng = np.random.default_rng(1000 + int(animal_id))
    traces = rng.normal(0.0, 0.08, size=(N_FRAMES, len(CELL_IDS)))

    for cell_idx, cell_id in enumerate(CELL_IDS):
        responsive_tones = patterns[cell_id]
        for tone_num, onset in enumerate(TONE_ONSETS, start=1):
            if tone_num in responsive_tones:
                start = onset + 1
                end = min(N_FRAMES, onset + 14)
                traces[start:end, cell_idx] += scale
                traces[end:min(N_FRAMES, end + 6), cell_idx] += scale * 0.35

        for bout_idx, (start, end) in enumerate(FREEZE_BOUTS):
            traces[start:min(N_FRAMES, start + 10), cell_idx] += 0.25 + 0.08 * ((cell_idx + bout_idx) % 3)
            traces[end:min(N_FRAMES, end + 6), cell_idx] -= 0.1

        traces[:, cell_idx] += np.linspace(-0.03, 0.03, N_FRAMES) * ((cell_idx % 2) * 2 - 1)

    if z_like:
        return traces.astype(np.float32)

    traces = traces + 1.5 + 0.05 * np.arange(len(CELL_IDS))
    return traces.astype(np.float32)


def write_zip_from_bytes(zip_path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload in members.items():
            zf.writestr(arcname, payload)


def make_ext1_zip() -> None:
    time, tone_flag, tone_id, freeze = build_common_flags()
    members: dict[str, bytes] = {}
    for animal_id in ANIMALS:
        z = make_signal(EXT1_PATTERNS[animal_id], animal_id, scale=1.8, z_like=True)
        buf = io.BytesIO()
        np.savez(
            buf,
            Time_s=time.astype(np.float32),
            Z=z,
            ToneFlag=tone_flag.astype(np.int16),
            FreezeFlag=freeze.astype(np.int16),
            ToneIndex=tone_id.astype(np.int16),
            cell_ids=np.array(CELL_IDS, dtype=str),
        )
        members[f"animal{animal_id}_ext1_zscored_traces.npz"] = buf.getvalue()
    write_zip_from_bytes(ROOT / "sample_ext1_zscored_npz_part1.zip", members)


def make_ret_npz_zip() -> None:
    time, tone_flag, tone_id, freeze = build_common_flags()
    members: dict[str, bytes] = {}
    for animal_id in ANIMALS:
        z = make_signal(RET_PATTERNS[animal_id], animal_id, scale=1.7, z_like=True)
        buf = io.BytesIO()
        np.savez(
            buf,
            time=time.astype(np.float32),
            z=z,
            tone_flag=tone_flag.astype(np.int16),
            freeze=freeze.astype(np.int16),
            tone_id=tone_id.astype(np.int16),
            cell_ids=np.array(CELL_IDS, dtype=str),
        )
        members[f"animal{animal_id}_retrieval_zscored_traces.npz"] = buf.getvalue()
    write_zip_from_bytes(ROOT / "sample_ret_zscored_npz_bundle.zip", members)


def make_raw_csv_zip(zip_name: str, patterns_by_animal: dict[str, dict[str, set[int]]]) -> None:
    time, tone_flag, tone_id, freeze = build_common_flags()
    members: dict[str, bytes] = {}
    for animal_id in ANIMALS:
        traces = make_signal(patterns_by_animal[animal_id], animal_id, scale=2.4, z_like=False)
        df = pd.DataFrame({
            "Time_s": time,
            "ToneFlag": tone_flag,
            "FreezeFlag": freeze,
            "CS": tone_id,
        })
        for idx, cell_id in enumerate(CELL_IDS):
            df[cell_id] = traces[:, idx]
        members[f"animal{animal_id}_{zip_name.replace('.zip', '')}.csv"] = df.to_csv(index=False).encode("utf-8")
    write_zip_from_bytes(ROOT / zip_name, members)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    make_ext1_zip()
    make_ret_npz_zip()
    make_raw_csv_zip("sample_ext2_raw_trace_files.zip", EXT2_PATTERNS)
    make_raw_csv_zip("sample_ret_raw_trace_files.zip", RET_PATTERNS)
    print(f"Wrote sample data to {ROOT}")


if __name__ == "__main__":
    main()
