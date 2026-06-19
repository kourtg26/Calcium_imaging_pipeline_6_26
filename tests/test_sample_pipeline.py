from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


class SamplePipelineSmokeTest(unittest.TestCase):
    def test_sample_pipeline_runner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="grin-sample-pipeline-") as tmpdir:
            out_dir = Path(tmpdir) / "outputs"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "sample_data/run_sample_pipeline.py"),
                    "--output-dir",
                    str(out_dir),
                    "--regenerate-data",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            expected = [
                out_dir / "ext1_classes_proportions_only.zip",
                out_dir / "ext2_classes_proportions_only.zip",
                out_dir / "ret_classes_proportions_only.zip",
                out_dir / "Ext1_toneBlock_summary_meanSEM.csv",
                out_dir / "Ext2_toneBlock_summary_meanSEM.csv",
                out_dir / "Ret_toneByTone_summary_meanSEM.csv",
                out_dir / "Ext1_Ext2_Ret_transition_counts_triple_cellWeighted.csv",
                out_dir / "heatmap/Ext2Sample_toneOnly_heatmap_matrix.npz",
                out_dir / "freezeOnset_deltaZ_trace_eventCounts_perAnimal.csv",
                out_dir / "ext1_ext2_transition/Ext1toExt2_transition_counts_cellWeighted.csv",
                out_dir / "ext2_ret_transition/Ext2toRet_transition_counts_cellWeighted.csv",
                out_dir / "sample_pipeline_manifest.json",
            ]
            missing = [str(path) for path in expected if not path.exists()]
            self.assertEqual([], missing, f"Missing expected sample outputs: {missing}")

            ext1 = pd.read_csv(out_dir / "Ext1_classes_proportions_perAnimal.csv")
            ext2 = pd.read_csv(out_dir / "Ext2_classes_proportions_perAnimal.csv")
            ret = pd.read_csv(out_dir / "Ret_classes_proportions_perAnimal.csv")

            def weighted_responsive(df: pd.DataFrame) -> float:
                responsive = 1.0 - df["p_Neither"]
                return float((responsive * df["n_cells"]).sum() / df["n_cells"].sum())

            ext1_resp = weighted_responsive(ext1)
            ext2_resp = weighted_responsive(ext2)
            ret_resp = weighted_responsive(ret)

            self.assertGreater(ext1_resp, ext2_resp)
            self.assertGreater(ret_resp, ext2_resp)


if __name__ == "__main__":
    unittest.main()
