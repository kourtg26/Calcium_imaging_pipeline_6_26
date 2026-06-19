from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
