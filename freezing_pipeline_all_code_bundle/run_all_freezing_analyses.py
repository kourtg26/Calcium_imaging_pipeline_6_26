import argparse, os
from utils_freezing import load_config
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    return ap.parse_args()

import subprocess, sys, os

SCRIPTS=[
 "01_freezeOnset_deltaZ_meanTraces_allSessions.py",
 "02_freezeOnset_deltaZ_meanTraces_responsiveOnly_allSessions.py",
 "03_freezeOnset_deltaZ_meanTraces_splitByClass_allSessions.py",
 "04_freezeOnset_vs_toneOnset_allSessions.py",
 "05_freezeOnset_deltaZ_meanByToneSubset_splitByClass.py",
 "06_freezeOnsetDeltaZ_meanTraces_byToneSubset_splitByClass_ANIMALWEIGHTED.py",
 "07_freezeOnsetDeltaZ_meanTraces_byToneSubset_splitByClass_CELLWEIGHTED.py",
 "08_freezeResponsive_overlapToneOnset_byClass.py",
]

def main(cfg_path):
    for s in SCRIPTS:
        print(f"Running {s} ...")
        ret=subprocess.call([sys.executable, s, "--config", cfg_path])
        if ret!=0:
            raise SystemExit(ret)
    print("Done.")

if __name__=="__main__":
    args=parse_args()
    main(args.config)
