#!/usr/bin/env python3
"""
Two-scale geodesic kernel vs MOND vs NFW on SPARC ROTMOD

Usage:
  python fit_geodesic_twoscale.py --rotmod Rotmod_LTG --dens_zip BulgeDiskDec_LTG.zip --out outdir

Notes:
- Uses ℓ1 = β1*h and ℓ2 = β2*h (h from .dens) and global (α, w, β1, β2).
- Batches kernel evaluations with cached theta grid and reuses radial quadrature per galaxy.
- Trains on 50% random split, evaluates on all. Adjust --train_frac as needed.
"""

import os, io, zipfile, glob, re, argparse, json, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages

# ---------------- CLI ----------------
p = argparse.ArgumentParser()
p.add_argument("--rotmod", required=True, help="Path to ROTMOD folder or zip")
p.add_argument("--dens_zip", required=True, help="SPARC BulgeDiskDec_LTG.zip path")
p.add_argument("--out", required=True, help="Output directory")
p.add_argument("--train_frac", type=float, default=0.5)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--nr", type=int, default=160)
p.add_argument("--nth", type=int, default=24)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

# --------------- Data I/O ---------------
def read_rotmod(path: str):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#;":
                continue
            parts = re.split(r"[,\s]+", s)
            vals = []
            ok = True
            for x in parts:
                try:
                    vals.append(float(x))
                except:
                    ok = False; break
            if ok and len(vals)>=3:
                rows.append(vals)
    if not rows: return None
    A = np.array(rows, float)
    r = A[:,0]; vobs=A[:,1]; dv=A[:,2]
    vgas  = A[:,3] if A.shape[1]>3 else None
    vdisk = A[:,4] if A.shape[1]>4 else None
    vbulg = A[:,5] if A.shape[1]>5 else None
    name = os.path.basename(path).replace("_rotmod.dat","")
    return dict(r=r, vobs=vobs, dv=dv, vgas=vgas, vdisk=vdisk, vbulg=vbulg, name=name)

def norm_name(n: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(n)).strip("_").upper()

# Load ROTMOD files
rot_files = []
if os.path.isdir(args.rotmod):
    rot_files = glob.glob(os.path.join(args.rotmod, "*_rotmod.dat"))
elif zipfile.is_zipfile(args.rotmod):
    tmpd = os.path.join(args.out, "_rot")
    os.makedirs(tmpd, exist_ok=True)
    with zipfile.ZipFile(args.rotmod,"r") as Z:
        for n in Z.namelist():
            if n.lower().endswith(".dat"):
                outp = os.path.join(tmpd, os.path.basename(n))
                with open(outp,"wb") as f: f.write(Z.read(n))
                rot_files.append(outp)
else:
    raise SystemExit("rotmod must be a folder or zip")

datasets = []
for pth in sorted(rot_files):
    d = read_rotmod(pth)
    if d is not None and len(d["r"])>=4 and np.nanmax(d["r"])>0:
        datasets.append(d)

# Extract h from .dens zip (robust parse for exponential disk scale)
def estimate_h_from_dens_zip(dens_zip):
    Z = zipfile.ZipFile(dens_zip,"r")
    names = Z.namelist()
    out = {}
    for nm in names:
        if not nm.lower().endswith(".dens"): continue
        try:
            raw = Z.read(nm).decode("utf-8","ignore").splitlines()
        except:
            continue
        rs=[]; sig=[]
        for line in raw:
            s=line.strip()
            if not s or s[0] in "#;": continue
            parts=re.split(r"[,\s]+", s)
            try:
                vals=[float(x) for x in parts]
            except:
                continue
            if len(vals)>=2:
                rs.append(vals[0]); sig.append(vals[1])
        if len(rs)<6: continue
        r=np.array(rs,float); y=np.array(sig,float)
        # fit ln(y) vs r over mid range (robust)
        msk=(y>0)&np.isfinite(y)&np.isfinite(r)
        if np.sum(msk)<6: continue
        r=r[msk]; y=y[msk]
        # choose middle 50% radii
        q1, q3 = np.percentile(r,[25,75])
        sel=(r>=q1)&(r<=q3)
        if np.sum(sel)<6: sel = np.ones_like(r, bool)
        rr=r[sel]; ln=np.log(y[sel])
        A=np.vstack([rr, np.ones_like(rr)]).T
        try:
            slope, intercept = np.linalg.lstsq(A, ln, rcond=None)[0]
            h = -1.0/slope if slope<0 else np.nan
        except Exception:
            h = np.nan
        key = os.path.basename(nm).replace(".dens","")
        out[norm_name(key)] = h/1000.0 if np.isfinite(h) else np.nan  # convert pc->kpc if needed
    return out

h_map = estimate_h_from_dens_zip(args.dens_zip)

def v_baryonic(r, d):
    v2 = np.zeros_like(r, dtype=float)
    if d["vgas"]  is not None:  v2 += np.interp(r, d["r"], d["vgas"])**2
    if d["vdisk"] is not None:  v2 += np.interp(r, d["r"], d["vdisk"])**2
    if d["vbulg"] is not None:  v2 += np.interp(r, d["r"], d["vbulg"])**2
    return np.sqrt(v2)

def a_bar(d, r=None):
    if r is None: r = d["r"]
    vbar = v_baryonic(r, d)
    rpos = np.where(r<=0, np.nan, r)
    return np.nan_to_num((vbar**2)/rpos, nan=0.0, posinf=0.0, neginf=0.0)

# Axisymmetric two-scale kernel (with caching of theta grid)
thetas = np.linspace(0, 2*np.pi, args.nth, endpoint=False)
cosT = np.cos(thetas)

def axisym_two_scale(d, ell1, ell2, w, nR=args.nr):
    r_data = d["r"]
    Rmax = float(np.nanmax(r_data))
    if not np.isfinite(Rmax) or Rmax <= 0:
        return np.zeros_like(r_data)
    Rg = np.linspace(0.0, max(Rmax, 5*max(ell1, ell2)), nR)
    abar_R = a_bar(d, r=Rg)
    dR = Rg[1]-Rg[0]
    inv1 = 1.0/(2*np.pi*ell1**2)
    inv2 = 1.0/(2*np.pi*ell2**2)
    out = np.zeros_like(r_data, dtype=float)
    rr_arr = r_data**2
    for i, r in enumerate(r_data):
        if r == 0:
            dists = Rg
            kernel = w*np.exp(-dists/ell1)*inv1 + (1-w)*np.exp(-dists/ell2)*inv2
            out[i] = np.trapz(kernel * abar_R * Rg * (2*np.pi), dx=dR)
        else:
            rr = rr_arr[i]
            theta_sum = np.zeros_like(Rg)
            for c in cosT:
                d = np.sqrt(rr + Rg*Rg - 2.0*r*Rg*c)
                theta_sum += w*np.exp(-d/ell1)*inv1 + (1-w)*np.exp(-d/ell2)*inv2
            theta_sum *= (2*np.pi/len(cosT))
            out[i] = np.trapz(theta_sum * abar_R * Rg, dx=dR)
    return out

def model_two(d, alpha, w, beta1, beta2):
    h = h_map.get(norm_name(d["name"]), np.nan)
    if not np.isfinite(h): h = 1.0
    ell1 = float(np.clip(beta1*h, 0.05, 40.0))
    ell2 = float(np.clip(beta2*h, 0.05, 40.0))
    w = float(np.clip(w, 0.0, 1.0))
    ageo = axisym_two_scale(d, ell1, ell2, w)
    atot = a_bar(d) + alpha*ageo
    return np.sqrt(np.maximum(d["r"]*atot, 0.0))

def chi2(d, v):
    return float(np.sum(((d["vobs"]-v)/np.maximum(d["dv"],1e-6))**2))

# Split
rng = np.random.default_rng(args.seed)
idx = np.arange(len(datasets)); rng.shuffle(idx)
ntr = max(1, int(args.train_frac*len(datasets)))
train = [datasets[i] for i in idx[:ntr]]
test  = [datasets[i] for i in idx[ntr:]]

# Objective on subset for speed
search = train[:60]

def objective(theta):
    a,w,b1,b2 = theta
    if not (0.0<=a<=2.0 and 0.0<=w<=1.0 and 0.05<=b1<=8.0 and 0.05<=b2<=8.0):
        return 1e12
    s=0.0
    for d in search:
        v = model_two(d, a,w,b1,b2)
        if not np.all(np.isfinite(v)): s += 1e6; continue
        s += chi2(d, v)
    return s

# Coarse seeds
alphas = np.linspace(0.4, 1.3, 5)
ws     = np.linspace(0.2, 0.8, 4)
betas1 = np.linspace(0.3, 1.6, 5)
betas2 = np.linspace(3.0, 6.5, 5)
best=None; best_val=np.inf
for a in alphas:
    for w0 in ws:
        for b1 in betas1:
            for b2 in betas2:
                val = objective((a,w0,b1,b2))
                if val < best_val:
                    best_val, best = val, (a,w0,b1,b2)

res = minimize(objective, x0=np.array(best), bounds=[(0.0,2.0),(0.0,1.0),(0.05,8.0),(0.05,8.0)],
               method="L-BFGS-B", options=dict(maxiter=80))
a,w,b1,b2 = (res.x if res.success else best)

# Evaluate on all
rows=[]
for d in datasets:
    v = model_two(d, a,w,b1,b2)
    c2 = chi2(d, v); red = c2/max(len(d["r"])-1,1)
    rms = float(np.sqrt(np.mean((d["vobs"]-v)**2)))
    rows.append(dict(name=d["name"], chi2=c2, red_chi2=red, rms=rms))
df_two = pd.DataFrame(rows).sort_values("red_chi2")
df_two.to_csv(os.path.join(args.out,"metrics_geodesic_twoscale_full.csv"), index=False)
pd.DataFrame(dict(alpha=[a], w=[w], beta1=[b1], beta2=[b2])).to_csv(os.path.join(args.out,"geodesic_twoscale_params.csv"), index=False)

print("DONE. Params:", a,w,b1,b2)
