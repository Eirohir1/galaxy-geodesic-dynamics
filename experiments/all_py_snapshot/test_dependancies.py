import os
import time
import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import Optional

# === SETTINGS ===
DATA_DIR = r"C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts"

# === CLASS ===
@dataclass
class RotmodData:
    r_kpc: np.ndarray
    v_obs: np.ndarray
    dv_obs: np.ndarray
    v_gas: Optional[np.ndarray] = None
    v_disk: Optional[np.ndarray] = None
    v_bulge: Optional[np.ndarray] = None

# === FILE LOADER ===
def read_rotmod(path: str) -> RotmodData:
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
                continue
            parts = s.replace(",", " ").split()
            try:
                vals = [float(x) for x in parts]
                rows.append(vals)
            except ValueError:
                continue
    arr = np.array(rows, dtype=float)
    r = arr[:, 0]
    vobs = arr[:, 1]
    dv = arr[:, 2]
    vgas = arr[:, 3] if arr.shape[1] > 3 else None
    vdisk = arr[:, 4] if arr.shape[1] > 4 else None
    vbulge = arr[:, 5] if arr.shape[1] > 5 else None
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulge)

# === GPU BENCHMARK ===
def benchmark_gpu():
    size = 5000
    print("\n=== GPU BENCHMARK ===")
    print("Running CPU test...")
    a_cpu = np.random.rand(size, size)
    b_cpu = np.random.rand(size, size)
    start_cpu = time.time()
    np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start_cpu
    print(f"CPU time: {cpu_time:.3f} seconds")

    print("\nRunning GPU test...")
    a_gpu = cp.random.rand(size, size)
    b_gpu = cp.random.rand(size, size)
    cp.cuda.Stream.null.synchronize()
    start_gpu = time.time()
    cp.dot(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start_gpu
    print(f"GPU time: {gpu_time:.3f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x faster with GPU")

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: DATA_DIR not found: {DATA_DIR}")
        exit(1)

    dat_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".dat")]
    if not dat_files:
        print("No .dat files found in the directory!")
        exit(1)

    print(f"Found {len(dat_files)} .dat files.")
    success_count = 0
    for fname in dat_files:
        path = os.path.join(DATA_DIR, fname)
        try:
            data = read_rotmod(path)
            print(f"[OK] {fname}: {len(data.r_kpc)} points, first radius={data.r_kpc[0]:.3f} kpc")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")

    print(f"\nLoaded {success_count}/{len(dat_files)} files successfully.")

    # Run GPU benchmark once
    benchmark_gpu()
