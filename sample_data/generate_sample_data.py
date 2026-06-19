#!/usr/bin/env python3
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
SAMPLE_ANIMALS = ["1", "2", "4", "6"]

EXT1_SOURCE_DIR = REPO_ROOT / "ext1_pipeline_output_2of3_3s_fullregen"
EXT2_SOURCE_ZIP = REPO_ROOT / "ext2_raw_trace_files.zip"
RET_ZSCORED_SOURCE_ZIP = REPO_ROOT / "retrieval_pipeline_outputs_bundle" / "ret_zscored_npz_bundle.zip"

OUT_EXT1_ZIP = ROOT / "sample_ext1_zscored_npz_part1.zip"
OUT_EXT2_ZIP = ROOT / "sample_ext2_raw_trace_files.zip"
OUT_RET_ZSCORED_ZIP = ROOT / "sample_ret_zscored_npz_bundle.zip"
OUT_RET_RAW_ZIP = ROOT / "sample_ret_raw_trace_files.zip"


def infer_animal_id(name: str) -> str | None:
    match = re.search(r"(?:animal)?(\d{1,5})", Path(name).name, flags=re.IGNORECASE)
    return match.group(1) if match else None


def write_zip_from_bytes(zip_path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload in members.items():
            zf.writestr(arcname, payload)


def collect_ext1_npzs() -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for animal_id in SAMPLE_ANIMALS:
        candidates = [
            EXT1_SOURCE_DIR / f"animal{animal_id}_ext1_zscored_traces.npz",
            EXT1_SOURCE_DIR / f"{animal_id}_ext1_zscored_traces.npz",
        ]
        src = next((path for path in candidates if path.exists()), None)
        if src is None:
            raise FileNotFoundError(f"Missing Ext1 source NPZ for animal {animal_id}")
        members[f"animal{animal_id}_ext1_zscored_traces.npz"] = src.read_bytes()
    return members


def subset_zip_members(src_zip: Path, suffix: str) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(src_zip, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(suffix):
                continue
            base_name = Path(name).name
            if base_name.startswith("._") or "__macosx" in name.lower():
                continue
            animal_id = infer_animal_id(name)
            if animal_id not in SAMPLE_ANIMALS:
                continue
            members[base_name] = zf.read(name)
    if len(members) != len(SAMPLE_ANIMALS):
        raise RuntimeError(f"Expected {len(SAMPLE_ANIMALS)} members from {src_zip.name}, found {len(members)}")
    return members


def collect_ret_zscored_npzs() -> dict[str, bytes]:
    return subset_zip_members(RET_ZSCORED_SOURCE_ZIP, ".npz")


def build_ret_raw_csvs(ret_zscored_members: dict[str, bytes]) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for arcname, payload in ret_zscored_members.items():
        animal_id = infer_animal_id(arcname)
        if animal_id is None:
            raise RuntimeError(f"Could not infer animal id from {arcname}")
        data = np.load(io.BytesIO(payload), allow_pickle=True)
        time = data["time"].astype(np.float32)
        z = data["z"].astype(np.float32)
        tone_flag = data["tone_flag"].astype(np.int16)
        freeze = data["freeze"].astype(np.int16)
        tone_id = data["tone_id"].astype(np.int16)
        cell_ids = [str(cid) for cid in data["cell_ids"]]

        meta_df = pd.DataFrame({
            "Time_s": time,
            "ToneFlag": tone_flag,
            "FreezeFlag": freeze,
            "CS": tone_id,
        })
        cell_df = pd.DataFrame(z, columns=cell_ids)
        df = pd.concat([meta_df, cell_df], axis=1)
        members[f"animal{animal_id}_retrieval_traces_from_real_subset.csv"] = df.to_csv(index=False).encode("utf-8")
    return members


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    ext1_members = collect_ext1_npzs()
    ext2_members = subset_zip_members(EXT2_SOURCE_ZIP, ".csv")
    ret_zscored_members = collect_ret_zscored_npzs()
    ret_raw_members = build_ret_raw_csvs(ret_zscored_members)

    write_zip_from_bytes(OUT_EXT1_ZIP, ext1_members)
    write_zip_from_bytes(OUT_EXT2_ZIP, ext2_members)
    write_zip_from_bytes(OUT_RET_ZSCORED_ZIP, ret_zscored_members)
    write_zip_from_bytes(OUT_RET_RAW_ZIP, ret_raw_members)

    print(f"Wrote real-data sample subset to {ROOT}")
    print(f"Animals: {', '.join(SAMPLE_ANIMALS)}")


if __name__ == "__main__":
    main()
