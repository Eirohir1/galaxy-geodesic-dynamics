# paths.py - centralizes data/output locations with relative defaults
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
# data directory can be overridden via env var SPARC_DATA_DIR
DATA_DIR = Path(os.environ.get("SPARC_DATA_DIR", BASE_DIR / "data" / "SPARC")).resolve()
# output directory
OUT_DIR = Path(os.environ.get("OUT_DIR", BASE_DIR / "out")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)