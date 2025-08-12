# Geodesic Reinforcement (Acceleration-Space, Normalized) — Global Fit on SPARC rotmod files
#
# This script:
# 1) Unpacks the user's Rotmod_LTG.zip
# 2) Reads all *_rotmod.dat files (radius [kpc], v_obs, dv, v_gas, v_disk, v_bulge)
# 3) Defines a physics-consistent model:
#       a_bar = v_bar^2 / r
#       a_geo = (K_ell * a_bar)(r) with normalized exponential kernel and area weighting w(r)=2πr
#       a_tot = a_bar + α a_geo
#       v_mod = sqrt(r * a_tot)
# 4) Fits global (α, ℓ) on a training split and evaluates on validation
# 5) Saves per-galaxy metrics CSV and a few example plots
#
# Files produced:
#  - /mnt/data/gr_fit_summary.csv
#  - /mnt/data/gr_train_val_report.txt
#  - /mnt/data/gr_example_fit_*.png (a few example galaxies)

import os, zipfile, glob, io, math, random, textwrap
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import pandas as pd

# ------------------------------
# 0) Unpack the uploaded archive
# ------------------------------
zip_path = "/mnt/data/Rotmod_LTG.zip"
extract_dir = "/mnt/data/rotmod_ltg"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

# -------------------------------------
# 1) Load rotmod files into data frames
# -------------------------------------
class RotmodData:
    def __init__(self, r_kpc, v_obs, dv_obs, v_gas=None, v_disk=None, v_bulge=None, name=""):
        self.r_kpc = np.asarray(r_kpc, float)
        self.v_obs = np.asarray(v_obs, float)
        self.dv_obs = np.asarray(dv_obs, float)
        self.v_gas = np.asarray(v_gas, float) if v_gas is not None else None
        self.v_disk = np.asarray(v_disk, float) if v_disk is not None else None
        self.v_bulge = np.asarray(v_bulge, float) if v_bulge is not None else None
        self.name = name

def read_rotmod(path: str):
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#;":
                continue
            parts = s.replace(",", " ").split()
            vals = []
            ok = True
            for x in parts:
                try:
                    vals.append(float(x))
                except ValueError:
                    ok = False
                    break
            if ok and len(vals) >= 3:
                rows.append(vals)
    if not rows:
        return None
    A = np.array(rows, float)
    r = A[:,0]; vobs=A[:,1]; dv=A[:,2]
    vgas  = A[:,3] if A.shape[1] > 3 else None
    vdisk = A[:,4] if A.shape[1] > 4 else None
    vbulg = A[:,5] if A.shape[1] > 5 else None
    name = os.path.basename(path).replace("_rotmod.dat","").replace(".dat","")
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulg, name)

def v_baryonic(r_kpc: np.ndarray, d: RotmodData) -> np.ndarray:
    r = np.asarray(r_kpc, float)
    v2 = np.zeros_like(r)
    if d.v_gas  is not None: v2 += np.interp(r, d.r_kpc, d.v_gas )**2
    if d.v_disk is not None: v2 += np.interp(r, d.r_kpc, d.v_disk)**2
    if d.v_bulge is not None: v2 += np.interp(r, d.r_kpc, d.v_bulge)**2
    return np.sqrt(np.maximum(v2, 0.0))

# ----------------------------------------------------
# 2) Geodesic model: acceleration-space, normalized Kℓ
# ----------------------------------------------------
def model_nonlocal_geo(d: RotmodData, alpha: float, ell: float, use_area_weight=True):
    r_data = np.asarray(d.r_kpc, float)
    # Guard: need at least 4 points and positive radii
    if len(r_data) < 4 or np.nanmax(r_data) <= 0:
        return None

    vbar = v_baryonic(r_data, d)
    # Avoid division by zero at r=0
    r_pos = np.where(r_data<=0, np.nan, r_data)
    a_bar = vbar**2 / r_pos
    # Replace infinities/NaNs from r=0
    if np.isnan(a_bar[0]):
        a_bar[0] = a_bar[1] if len(a_bar) > 1 else 0.0
    a_bar = np.nan_to_num(a_bar, nan=0.0, posinf=0.0, neginf=0.0)

    # Fine grid for stable convolution
    Rmax = float(np.nanmax(r_data))
    dr_native = np.median(np.diff(np.unique(np.round(r_data,6)))) if len(r_data)>1 else 0.1
    dr = max(0.05, min(0.2, dr_native))  # 0.05–0.2 kpc
    R = np.arange(0.0, Rmax + 5.0*ell, dr)
    if len(R) < 10:
        R = np.linspace(0, max(Rmax, 5.0*ell), 200)

    abar_f = np.interp(R, r_data, a_bar)

    # Weight: axisymmetric area measure
    W = (2.0*np.pi*R) if use_area_weight else np.ones_like(R)

    # Exponential kernel on R (distance from 0); we use convolution with symmetric assumption
    K = np.exp(-R/ell)
    # Normalize under weight: ∫ K * W dR = 1
    norm = np.trapz(K * W, R)
    if not np.isfinite(norm) or norm <= 0:
        return None
    K /= norm

    # Convolution: (a_bar * W) convolved with K, then multiply by dr
    conv = fftconvolve(abar_f * W, K, mode="same") * (R[1]-R[0])

    a_tot_f = abar_f + alpha * conv
    a_tot = np.interp(r_data, R, a_tot_f)
    v_mod = np.sqrt(np.maximum(r_data * a_tot, 0.0))
    return v_mod

def chi2_galaxy(d: RotmodData, v_model: np.ndarray):
    if v_model is None: 
        return np.inf, np.inf
    resid = (d.v_obs - v_model) / np.maximum(d.dv_obs, 1e-6)
    chi2 = float(np.sum(resid**2))
    red = chi2 / max(len(d.v_obs)-1, 1)
    return chi2, red

# --------------------------
# 3) Assemble the data set
# --------------------------
files = sorted(glob.glob(os.path.join(extract_dir, "**", "*_rotmod.dat"), recursive=True))
datasets = []
for p in files:
    d = read_rotmod(p)
    if d is None: 
        continue
    # Minimal sanity filters
    if np.any(~np.isfinite(d.r_kpc)) or np.nanmax(d.r_kpc)<=0:
        continue
    if len(d.r_kpc) < 4:
        continue
    datasets.append(d)

N = len(datasets)

# --------------------------
# 4) Train/validation split
# --------------------------
random.seed(42)
idx = list(range(N))
random.shuffle(idx)
train_count = max(1, int(0.33 * N))
train_idx = set(idx[:train_count])
val_idx   = set(idx[train_count:])

train_set = [datasets[i] for i in train_idx]
val_set   = [datasets[i] for i in val_idx]

# ----------------------------------------------------
# 5) Global fit of (alpha, ell) on the training set
#    Bounds: alpha ∈ [0, 2], ell ∈ [0.2, 20] kpc
# ----------------------------------------------------
def objective_theta(theta, ds_list):
    alpha, ell = theta
    if not (0.0 <= alpha <= 2.0 and 0.1 <= ell <= 40.0):
        return 1e12
    total = 0.0
    for d in ds_list:
        v = model_nonlocal_geo(d, alpha, ell, use_area_weight=True)
        if v is None or not np.all(np.isfinite(v)):
            total += 1e6
            continue
        c2, _ = chi2_galaxy(d, v)
        total += c2
    return total

# Coarse grid search to seed the optimizer
alphas = np.linspace(0.1, 1.5, 15)
ells   = np.linspace(0.5, 20.0, 20)
best = (0.5, 5.0); best_val = float("inf")
for a in alphas:
    for e in ells:
        val = objective_theta((a,e), train_set)
        if val < best_val:
            best_val = val
            best = (a,e)

# Local refinement
res = minimize(lambda th: objective_theta(th, train_set),
               x0=np.array(best),
               bounds=[(0.0,2.0),(0.1,40.0)],
               method="L-BFGS-B",
               options=dict(maxiter=200))
alpha_star, ell_star = (res.x if res.success else best)

# -----------------------------------------
# 6) Evaluate on train and validation sets
# -----------------------------------------
def eval_set(ds_list, alpha, ell, tag):
    rows = []
    for d in ds_list:
        v = model_nonlocal_geo(d, alpha, ell, use_area_weight=True)
        c2, red = chi2_galaxy(d, v)
        rows.append(dict(name=d.name, n=len(d.r_kpc), chi2=c2, red_chi2=red))
    df = pd.DataFrame(rows).sort_values("red_chi2")
    return df

df_train = eval_set(train_set, alpha_star, ell_star, "train")
df_val   = eval_set(val_set,   alpha_star, ell_star, "val")

# --------------------------------------
# 7) Save report, CSV, and example plots
# --------------------------------------
report_lines = []
report_lines.append("Global fit of Geodesic Reinforcement (acceleration-space, normalized, area-weighted)")
report_lines.append(f"Files: {N}  (train={len(train_set)}, val={len(val_set)})")
report_lines.append(f"alpha* = {alpha_star:.4f},  ell* = {ell_star:.3f} kpc")
report_lines.append("")
for tag, df in [("TRAIN", df_train), ("VALID", df_val)]:
    med = df["red_chi2"].median()
    p90 = df["red_chi2"].quantile(0.9)
    report_lines.append(f"{tag}: median red-chi2 = {med:.2f},  90th pct = {p90:.2f}")
report_txt = "\n".join(report_lines)

with open("/mnt/data/gr_train_val_report.txt","w") as f:
    f.write(report_txt)

# Save CSV of per-galaxy metrics
df_all = pd.concat([df_train.assign(split="train"), df_val.assign(split="val")], ignore_index=True)
df_all.to_csv("/mnt/data/gr_fit_summary.csv", index=False)

# Generate a few example plots from validation set (best/median/worst by red chi2)
def plot_example(d: RotmodData, alpha, ell, out_path):
    r = d.r_kpc
    vbar = v_baryonic(r, d)
    vmod = model_nonlocal_geo(d, alpha, ell, use_area_weight=True)

    plt.figure(figsize=(6,4))
    plt.title(f"{d.name}  (α={alpha:.3f}, ℓ={ell:.2f} kpc)")
    # obs ± σ
    plt.fill_between(r, d.v_obs - d.dv_obs, d.v_obs + d.dv_obs, alpha=0.2, label="obs ±σ")
    plt.plot(r, d.v_obs, lw=1.8, label="v_obs")
    plt.plot(r, vbar, lw=1.4, label="v_baryon")
    plt.plot(r, vmod, lw=2.0, label="model")
    plt.xlabel("r [kpc]"); plt.ylabel("v [km/s]")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

# Choose examples
examples = []
if not df_val.empty:
    examples.append(df_val.iloc[0]["name"])  # best
    examples.append(df_val.iloc[len(df_val)//2]["name"])  # median-ish
    examples.append(df_val.iloc[-1]["name"])  # worst

name_to_data = {d.name: d for d in datasets}
saved_plots = []
for i, nm in enumerate(examples):
    d = name_to_data.get(nm)
    if d is None: 
        continue
    outp = f"/mnt/data/gr_example_fit_{i+1}_{nm}.png"
    plot_example(d, alpha_star, ell_star, outp)
    saved_plots.append(outp)

# Display dataframes for the user
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("Geodesic Reinforcement — Train Set (per-galaxy)", df_train)
display_dataframe_to_user("Geodesic Reinforcement — Validation Set (per-galaxy)", df_val)

report_txt, "/mnt/data/gr_train_val_report.txt", "/mnt/data/gr_fit_summary.csv", saved_plots
