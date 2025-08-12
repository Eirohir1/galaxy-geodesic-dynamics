#!/usr/bin/env python3
"""
get_sparc_metadata.py  (patched robust)
- Downloads SPARC Bulge–Disk Decompositions zip
- Tries multiple parsing strategies to extract disk scale lengths (h) [kpc]:
  1) Per-galaxy TXT/DAT files: lines mentioning 'scale length' or 'h = ... kpc'
  2) Any .mrt/.tab master table with columns containing 'h' or 'scale' and 'kpc'
- Writes:
  - sparc_h.csv (name, h_kpc, source, name_raw)
  - _sparc_cache/zip_filelist.txt (listing of ZIP contents for debugging)
"""

import os, io, re, sys, zipfile
from typing import List, Dict, Optional
import requests
import pandas as pd

SPARC_HOME = "https://astroweb.case.edu/SPARC/"
BULGE_DISK_ZIP = "BulgeDiskDec_LTG.zip"

def http_get(url: str, timeout=60) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def try_download_bulgedisk(out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    url = SPARC_HOME + BULGE_DISK_ZIP
    print(f"Downloading: {url}")
    data = http_get(url)
    zpath = os.path.join(out_dir, BULGE_DISK_ZIP)
    with open(zpath, "wb") as f:
        f.write(data)
    print(f"Wrote: {zpath}  ({len(data)/1e6:.2f} MB)")
    return zpath

def norm_name(n: str) -> str:
    n2 = re.sub(r"[^A-Za-z0-9]+", "_", str(n)).strip("_")
    return n2.upper()

def parse_per_file_texts(Z: zipfile.ZipFile, names: List[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        try:
            raw = Z.read(name).decode("utf-8", "ignore")
        except Exception:
            continue
        h = None
        # Scan for lines with scale length in kpc
        for line in raw.splitlines():
            s = line.strip().lower()
            if ("scale length" in s or re.match(r"^h\s*=", s)) and "kpc" in s:
                m = re.search(r"([-+]?\d+(\.\d+)?)", s)
                if m:
                    h = float(m.group(1)); break
        galaxy = os.path.basename(name).rsplit(".",1)[0]
        rows.append({"name": norm_name(galaxy), "h_kpc": h, "source": "per-file", "name_raw": galaxy})
    return pd.DataFrame(rows)

def parse_mrt_tables(Z: zipfile.ZipFile, names: List[str]) -> pd.DataFrame:
    # Try to parse an .mrt/.tab that may have a column containing disk scale length.
    # We don't know the exact column name; try candidates like 'h', 'Rd', 'R_d', 'scale'.
    frames = []
    for name in names:
        try:
            raw = Z.read(name).decode("utf-8", "ignore")
        except Exception:
            continue
        # Heuristic parse: split on lines, find header row
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        # Find a line that looks like a header with multiple column names
        header_idx = None
        for i, ln in enumerate(lines[:50]):
            # crude heuristic: commas or multiple spaces indicating columns
            if ("," in ln and len(ln.split(","))>=3) or (len(re.split(r"\s{2,}", ln))>=3):
                header_idx = i; break
        if header_idx is None:
            continue
        header_line = lines[header_idx]
        # Tokenize header
        if "," in header_line:
            cols = [c.strip() for c in header_line.split(",")]
            delim = ","
        else:
            cols = [c.strip() for c in re.split(r"\s{2,}", header_line) if c.strip()]
            delim = None
        # Build rows until a blank or separator-like line
        data_rows = []
        for ln in lines[header_idx+1:]:
            if set(ln) <= set("-=+_*") or len(ln)<3:
                break
            if delim == ",":
                toks = [t.strip() for t in ln.split(",")]
            else:
                toks = [t.strip() for t in re.split(r"\s{2,}", ln) if t.strip()]
            if len(toks) < len(cols):
                # pad
                toks = toks + [""]*(len(cols)-len(toks))
            data_rows.append(toks[:len(cols)])
        if not data_rows:
            continue
        try:
            df = pd.DataFrame(data_rows, columns=cols)
        except Exception:
            # fallback with generic columns
            df = pd.DataFrame(data_rows)
        df["_source_file"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["name","h_kpc","source","name_raw"])
    big = pd.concat(frames, ignore_index=True, sort=False)
    # Try to find a name column
    name_col = None
    for cand in ["Name","name","Galaxy","GALAXY","Object","ID"]:
        if cand in big.columns:
            name_col = cand; break
    if name_col is None:
        # try first column
        name_col = big.columns[0]
    # Try to find an h-like column
    h_col = None
    for cand in big.columns:
        cl = str(cand).lower()
        if any(k in cl for k in ["scale", "rd", "r_d", "h"]):
            h_col = cand; break
    if h_col is None:
        # give up
        return pd.DataFrame(columns=["name","h_kpc","source","name_raw"])
    # Clean numeric
    def to_float(x):
        try:
            return float(str(x).split()[0])
        except Exception:
            return float("nan")
    out = pd.DataFrame({
        "name_raw": big[name_col].astype(str),
        "h_kpc": big[h_col].map(to_float)
    })
    out["name"] = out["name_raw"].map(norm_name)
    out["source"] = "mrt"
    # Drop obvious empties/dupes
    out = (out.dropna(subset=["h_kpc"])
              .sort_values(["name","h_kpc"], ascending=[True,False])
              .drop_duplicates("name", keep="first"))
    return out[["name","h_kpc","source","name_raw"]]

def main():
    out_dir = os.path.abspath("./_sparc_cache")
    os.makedirs(out_dir, exist_ok=True)
    zpath = try_download_bulgedisk(out_dir)
    with zipfile.ZipFile(zpath, "r") as Z:
        names_all = Z.namelist()
        # Write file list for debugging
        with open(os.path.join(out_dir, "zip_filelist.txt"), "w", encoding="utf-8") as f:
            for nm in names_all:
                f.write(nm+"\n")
        print(f"[INFO] ZIP contains {len(names_all)} files. Wrote zip_filelist.txt")
        # Candidate files by type
        text_like = [n for n in names_all if n.lower().endswith((".txt",".dat",".tab",".mrt",".csv"))]
        per_file = [n for n in text_like if "/LTG/" in n or "/ltg/" in n or n.lower().endswith((".txt",".dat"))]
        mrt_like = [n for n in text_like if n.lower().endswith((".mrt",".tab",".csv"))]
        # Strategy 1: per-galaxy files
        df1 = parse_per_file_texts(Z, per_file) if per_file else pd.DataFrame(columns=["name","h_kpc","source","name_raw"])
        # Strategy 2: MRT-like tables
        df2 = parse_mrt_tables(Z, mrt_like) if mrt_like else pd.DataFrame(columns=["name","h_kpc","source","name_raw"])
        df = pd.concat([df1, df2], ignore_index=True, sort=False)
        if df.empty:
            print("[WARN] Could not parse any h values from the ZIP. See _sparc_cache/zip_filelist.txt")
        else:
            df = (df.sort_values(["name","h_kpc"], ascending=[True,False])
                    .drop_duplicates("name", keep="first"))
            df.to_csv("sparc_h.csv", index=False)
            print(f"[OK] Wrote sparc_h.csv with {len(df)} rows.")

if __name__ == "__main__":
    main()
