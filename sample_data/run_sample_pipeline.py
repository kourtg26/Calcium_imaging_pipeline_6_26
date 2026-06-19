#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = Path(__file__).resolve().parent
REQUIRED_ARCHIVES = [
    SAMPLE_DIR / "sample_ext1_zscored_npz_part1.zip",
    SAMPLE_DIR / "sample_ext2_raw_trace_files.zip",
    SAMPLE_DIR / "sample_ret_raw_trace_files.zip",
    SAMPLE_DIR / "sample_ret_zscored_npz_bundle.zip",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the GRIN sample dataset through the main Python pipelines.")
    ap.add_argument("--output-dir", required=True, help="Directory where sample pipeline outputs should be written.")
    ap.add_argument("--regenerate-data", action="store_true", help="Rebuild the sample input archives before running.")
    return ap.parse_args()


def render_json_template(src: Path, data_dir: Path, output_dir: Path) -> Path:
    rendered = src.read_text(encoding="utf-8")
    rendered = rendered.replace("__DATA_DIR__", str(data_dir))
    rendered = rendered.replace("__OUTPUT_DIR__", str(output_dir))
    dest = output_dir / src.name.replace(".sample", ".resolved")
    dest.write_text(rendered, encoding="utf-8")
    return dest


def render_yaml_template(src: Path, data_dir: Path, output_dir: Path) -> Path:
    rendered = src.read_text(encoding="utf-8")
    rendered = rendered.replace("__DATA_DIR__", str(data_dir))
    rendered = rendered.replace("__OUTPUT_DIR__", str(output_dir))
    dest = output_dir / src.name.replace(".sample", ".resolved")
    dest.write_text(rendered, encoding="utf-8")
    return dest


def run(cmd: list[str], cwd: Path | None = None) -> None:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) if not current else f"{REPO_ROOT}{os.pathsep}{current}"
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".tmp_mplconfig"))
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else str(REPO_ROOT), env=env, check=True)


def ensure_sample_archives(force: bool) -> None:
    if force or any(not path.exists() for path in REQUIRED_ARCHIVES):
        run([sys.executable, str(SAMPLE_DIR / "generate_sample_data.py")], cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / ".tmp_mplconfig").mkdir(parents=True, exist_ok=True)

    ensure_sample_archives(force=args.regenerate_data)

    config_path = render_json_template(SAMPLE_DIR / "pipeline_config.sample.json", SAMPLE_DIR, output_dir)
    heatmap_cfg = render_yaml_template(SAMPLE_DIR / "heatmap_config.sample.yaml", SAMPLE_DIR, output_dir)

    common = [sys.executable]
    run(common + [str(REPO_ROOT / "ext1_pipeline_all_code_bundle/00_ext1_process_traces.py"), "--config", str(config_path)])
    run(common + [str(REPO_ROOT / "ext1_pipeline_all_code_bundle/01_ext1_tone_by_tone_activity_freeze.py"), "--config", str(config_path)])
    run(common + [str(REPO_ROOT / "ext2_pipeline_all_code_bundle/00_ext2_process_raw_traces.py"), "--config", str(config_path)])
    run(common + [str(REPO_ROOT / "ext2_pipeline_all_code_bundle/01_ext2_tone_by_tone_activity_freeze.py"), "--config", str(config_path)])
    run(common + [str(REPO_ROOT / "retrieval_pipeline_all_code_bundle/00_retrieval_process_raw_traces.py"), "--config", str(config_path)])
    run(common + [str(REPO_ROOT / "retrieval_pipeline_all_code_bundle/01_retrieval_tone_by_tone_activity_freeze.py"), "--config", str(config_path)])
    run(common + [str(REPO_ROOT / "retrieval_pipeline_all_code_bundle/02_ext1_ext2_ret_transitions_alluvial.py"), "--config", str(config_path)])
    run(common + [str(REPO_ROOT / "heatmap_pipeline_cli_config_bundle/make_heatmap.py"), "--config", str(heatmap_cfg)])

    run(
        common
        + [
            str(REPO_ROOT / "ext1_ext2_transition_pipeline_bundle/ext1_ext2_transition_pipeline.py"),
            "--ext1_zip",
            str(output_dir / "ext1_classes_proportions_only.zip"),
            "--ext2_raw_zips",
            str(SAMPLE_DIR / "sample_ext2_raw_trace_files.zip"),
            "--out_dir",
            str(output_dir / "ext1_ext2_transition"),
            "--work_dir",
            str(output_dir / "_ext1_ext2_transition_work"),
            "--early_n",
            "2",
            "--late_n",
            "2",
            "--min_hits_per_period",
            "1",
        ]
    )
    run(
        common
        + [
            str(REPO_ROOT / "ext2_ret_transition_pipeline_bundle/ext2_ret_transition_pipeline.py"),
            "--ret_zip",
            str(output_dir / "ret_classes_proportions_only.zip"),
            "--ext2_raw_zips",
            str(SAMPLE_DIR / "sample_ext2_raw_trace_files.zip"),
            "--out_dir",
            str(output_dir / "ext2_ret_transition"),
            "--work_dir",
            str(output_dir / "_ext2_ret_transition_work"),
            "--early_n",
            "2",
            "--late_n",
            "2",
            "--min_hits_per_period",
            "1",
        ]
    )

    run(
        common + ["run_all_freezing_analyses.py", "--config", str(config_path)],
        cwd=REPO_ROOT / "freezing_pipeline_all_code_bundle",
    )

    manifest = {
        "config": str(config_path),
        "heatmap_config": str(heatmap_cfg),
        "output_dir": str(output_dir),
    }
    (output_dir / "sample_pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Sample pipeline run complete: {output_dir}")


if __name__ == "__main__":
    main()
