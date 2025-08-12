# Reproduction Package (v3)
Updated: 2025-08-12

## What this is
Minimal, anonymized supplement to reproduce the main figures and diagnostics for the geodesic analysis.
Runs on CPU; GPU is optional.

## Quick start (CPU)
```
python -m venv .venv
.venv\Scripts\activate  # (Windows)
# or: source .venv/bin/activate  # (macOS/Linux)
pip install -r requirements.txt
# Prepare SPARC data (download zip manually; see fetch_sparc.py)
python fetch_sparc.py --zip /path/to/SPARC.zip --out data/SPARC
# or place your SPARC folders under data/SPARC manually
python run_all.py
```
Figures and CSVs will appear in `out/`.

## Data layout expected
```
data/SPARC/
  Rotmod_LTG/...
  RCs/...
  MD/...
```

## Config & paths
- Relative by default (`paths.py`).
- Override with env vars:
  - `SPARC_DATA_DIR=...`
  - `OUT_DIR=...`

## GPU (optional)
See `GPU_SETUP.md`.

## Requirements
- Python 3.11
- Packages: numpy, scipy, pandas, matplotlib, astropy (plus optional cupy-cuda12x, numba)

## License & citation
- SPARC dataset is by Lelli, McGaugh & Schombert (2016). Please cite the original paper when using the data.
- We do not redistribute SPARC files here; fetch locally with `fetch_sparc.py`.