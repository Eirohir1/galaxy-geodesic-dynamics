import os
import numpy as np
import pandas as pd
from typing import Optional

DATA_DIR = r"C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts"

def read_rotmod(path: str) -> Optional[dict]:
    rows = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("#", ";")):
                    continue
                parts = line.replace(",", " ").split()
                try:
                    values = list(map(float, parts[:6]))  # max 6 columns
                    rows.append(values)
                except ValueError:
                    continue
        if not rows:
            return None
        arr = np.array(rows)
        return {
            "r_kpc": arr[:, 0],
            "v_obs": arr[:, 1],
            "dv_obs": arr[:, 2],
            "v_gas": arr[:, 3] if arr.shape[1] > 3 else None,
            "v_disk": arr[:, 4] if arr.shape[1] > 4 else None,
            "v_bulge": arr[:, 5] if arr.shape[1] > 5 else None
        }
    except Exception:
        return None

def infer_galaxy_type(data: dict) -> str:
    r = data["r_kpc"]
    v = data["v_obs"]
    vmax = np.max(v)
    rmax = r[np.argmax(v)]
    extent = np.max(r)

    steepness = (vmax / extent) if extent > 0 else 0
    has_gas = data["v_gas"] is not None
    has_disk = data["v_disk"] is not None
    has_bulge = data["v_bulge"] is not None

    if vmax < 50 and extent < 5:
        return "dwarf"
    elif vmax > 200 and extent > 15:
        return "super-spiral"
    elif steepness < 4 and not has_bulge and not has_disk:
        return "diffuse"
    elif 50 <= vmax <= 200 and extent >= 5:
        return "spiral"
    else:
        return "uncertain"

def classify_all_galaxies(data_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(data_dir) if f.endswith("_rotmod.dat")]
    results = []
    for fname in sorted(files):
        fpath = os.path.join(data_dir, fname)
        data = read_rotmod(fpath)
        if data is None:
            results.append((fname, None, None, None, "unreadable"))
            continue
        vmax = np.max(data["v_obs"])
        rmax = data["r_kpc"][np.argmax(data["v_obs"])]
        gtype = infer_galaxy_type(data)
        results.append((fname, vmax, rmax, np.max(data["r_kpc"]), gtype))
    df = pd.DataFrame(results, columns=["Filename", "Vmax", "Rmax", "Extent_kpc", "GalaxyType"])
    return df

if __name__ == "__main__":
    df = classify_all_galaxies(DATA_DIR)
    out_path = os.path.join(DATA_DIR, "classified_galaxies.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Classification complete. Results saved to:\n{out_path}")
    print(df["GalaxyType"].value_counts())
