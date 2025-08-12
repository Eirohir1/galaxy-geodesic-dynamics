# run_all.py - Minimal end-to-end reproduction runner.
# - Assumes SPARC data are located under ./data/SPARC (see fetch_sparc.py).
# - Writes all figures/results to ./out by default.

import subprocess, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd):
    print('>>>', ' '.join(cmd))
    r = subprocess.run(cmd, cwd=str(BASE_DIR))
    if r.returncode != 0:
        sys.exit(r.returncode)

# 1) Kernel diagnostics figure
if (BASE_DIR / "multi_kernel_diagnostics.py").exists():
    run([sys.executable, "multi_kernel_diagnostics.py"])

# 2) Multi-kernel analysis (summary bar chart)
if (BASE_DIR / "honest_multikernel_analysis.py").exists():
    run([sys.executable, "honest_multikernel_analysis.py"])

print("\nDone. Check the 'out/' folder for figures and CSVs.")