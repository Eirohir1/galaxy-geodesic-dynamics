# fetch_sparc.py - Helper to place SPARC rotation-curve files locally.
#
# Usage:
#   1) Manually download the SPARC dataset ZIPs (Lelli+ 2016). Search:
#      'SPARC rotation curves data Lelli 2016'.
#   2) Put the downloaded ZIP(s) in the same folder as this script, then run:
#      python fetch_sparc.py --zip SPARC_LTG.zip --out data/SPARC
#   3) After extraction, your structure should be similar to:
#      data/SPARC/Rotmod_LTG/*.dat
#      data/SPARC/RCs/*.dat
#      data/SPARC/MD/*.dat
#
# Notes:
#   - This script does NOT download from the internet to preserve anonymity.
#   - You can also set SPARC_DATA_DIR environment variable to point to your SPARC folder.

import argparse, zipfile
from pathlib import Path

def extract_zip(zip_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    print(f"Extracted {zip_path} -> {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, help="Path to a SPARC ZIP you downloaded manually.")
    p.add_argument("--out", default="data/SPARC", help="Destination directory (relative ok).")
    args = p.parse_args()
    extract_zip(args.zip, args.out)