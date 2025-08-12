#!/usr/bin/env python3
"""
Two-scale geodesic kernel vs MOND vs NFW on SPARC ROTMOD — GPU-accelerated

Usage (examples):
  python fit_geodesic_twoscale_gpu.py --rotmod Rotmod_LTG --dens_zip BulgeDiskDec_LTG.zip --out outdir
  python fit_geodesic_twoscale_gpu.py --rotmod Rotmod_LTG.zip --dens_zip BulgeDiskDec_LTG.zip --out outdir --nr 200 --nth 32

Notes:
- Uses ℓ1 = β1*h and ℓ2 = β2*h (h from .dens) and global (α, w, β1, β2).
- If CuPy is available, θ–R integration runs on the GPU; otherwise falls back to NumPy.
- Trains on 50% random split, evaluates on all. Adjust --train_frac as needed.
"""

import os, zipfile, glob, re, argparse, math
import numpy as np, pandas as pd
from scipy.optimize import minimize

# ---------------- GPU backend ----------------
USE_GPU = False
try:
    import cupy as cp
    from cupyx.scipy import integrate as cpx_integrate
    USE_GPU = True
except Exception:
    cp = None
    USE_GPU = False

def get_xp():
    return cp if USE_GPU else np

def to_xp(a):
    if USE_GPU:
        return cp.asarray(a)
    return np.asarray(a)

def to_np(a):
    try:
        return cp.asnumpy(a) if USE_GPU else np.asarray(a)
    except Exception:
        return np.asarray(a)

def trapz_xp(y, x=None, dx=1.0, axis=-1):
    # CuPy's integrate.trapz mirrors NumPy; use dx when x is None
    if USE_GPU:
        return cpx_integrate.trapz(y, x=x, dx=dx, axis=axis)
    else:
        return np.trapz(y, x=x, dx=dx, axis=axis)

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

# ---------------- Data I/O ----------------
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
        msk=(y>0)&np.isfinite(y)&np.isfinite(r)
        if np.sum(msk)<6: continue
        r=r[msk]; y=y[msk]
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
        out[norm_name(key)] = h/1000.0 if np.isfinite(h) else np.nan  # pc→kpc if needed
    return out

h_map = estimate_h_from_dens_zip(args.dens_zip)

# ---------------- Physics helpers ----------------
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

# ---------------- GPU axisymmetric two-scale kernel ----------------
def axisym_two_scale_vec(d, ell1, ell2, w, nR, nth):
    """
    Vectorized θ–R integration. If GPU is enabled, all heavy ops run on the device.
    """
    xp = get_xp()
    r_data = to_xp(d["r"])
    Rmax = float(np.nanmax(d["r"]))
    if not np.isfinite(Rmax) or Rmax <= 0:
        return to_np(xp.zeros_like(r_data))

    # Build R grid and baryon accel on xp
    Rg = xp.linspace(0.0, max(Rmax, 5*max(ell1, ell2)), nR, dtype=xp.float64)
    abar_R = to_xp(a_bar(d, r=to_np(Rg)))  # a_bar uses numpy interp; pass host array

    inv1 = 1.0/(2*np.pi*ell1**2)
    inv2 = 1.0/(2*np.pi*ell2**2)

    thetas = xp.linspace(0, 2*xp.pi, nth, endpoint=False, dtype=xp.float64)
    cosT = xp.cos(thetas)[:, None]                  # shape (nth,1)
    Rg2  = Rg[None, :]                              # shape (1,nR)

    out = xp.zeros_like(r_data, dtype=xp.float64)
    dR  = (Rg[1]-Rg[0]).astype(xp.float64)

    for i in range(r_data.shape[0]):
        r = r_data[i].astype(xp.float64)
        if r == 0:
            dists = Rg
            kern = w*xp.exp(-dists/ell1)*inv1 + (1-w)*xp.exp(-dists/ell2)*inv2
            integrand = kern * abar_R * Rg * (2*xp.pi)
            out[i] = trapz_xp(integrand, dx=dR)
        else:
            rr = r*r
            # dists shape: (nth, nR)
            dists = xp.sqrt(rr + Rg2*Rg2 - 2.0*r*Rg2*cosT)
            k1 = xp.exp(-dists/ell1)*inv1
            k2 = xp.exp(-dists/ell2)*inv2
            kernel = w*k1 + (1-w)*k2
            theta_avg = kernel.mean(axis=0) * (2*xp.pi)  # average over θ then multiply by 2π
            integrand = theta_avg * abar_R * Rg
            out[i] = trapz_xp(integrand, dx=dR)
    return to_np(out)

def model_two(d, alpha, w, beta1, beta2, nR, nth):
    h = h_map.get(norm_name(d["name"]), np.nan)
    if not np.isfinite(h): h = 1.0
    ell1 = float(np.clip(beta1*h, 0.05, 40.0))
    ell2 = float(np.clip(beta2*h, 0.05, 40.0))
    w = float(np.clip(w, 0.0, 1.0))
    ageo = axisym_two_scale_vec(d, ell1, ell2, w, nR, nth)
    atot = a_bar(d) + alpha*ageo
    v = np.sqrt(np.maximum(d["r"]*atot, 0.0))
    return v

def chi2(d, v):
    return float(np.sum(((d["vobs"]-v)/np.maximum(d["dv"],1e-6))**2))

# ---------------- Split & objective ----------------
rng = np.random.default_rng(args.seed)
idx = np.arange(len(datasets)); rng.shuffle(idx)
ntr = max(1, int(args.train_frac*len(datasets)))
train = [datasets[i] for i in idx[:ntr]]
test  = [datasets[i] for i in idx[ntr:]]

search = train[:60]

def objective(theta):
    a,w,b1,b2 = theta
    if not (0.0<=a<=2.0 and 0.0<=w<=1.0 and 0.05<=b1<=8.0 and 0.05<=b2<=8.0):
        return 1e12
    s=0.0
    for d in search:
        v = model_two(d, a,w,b1,b2, args.nr, args.nth)
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

# ---------------- Evaluate on all ----------------
rows=[]
for d in datasets:
    v = model_two(d, a,w,b1,b2, args.nr, args.nth)
    c2 = chi2(d, v); red = c2/max(len(d["r"])-1,1)
    rms = float(np.sqrt(np.mean((d["vobs"]-v)**2)))
    rows.append(dict(name=d["name"], chi2=c2, red_chi2=red, rms=rms))
df_two = pd.DataFrame(rows).sort_values("red_chi2")
df_two.to_csv(os.path.join(args.out,"metrics_geodesic_twoscale_full.csv"), index=False)
pd.DataFrame(dict(alpha=[a], w=[w], beta1=[b1], beta2=[b2], gpu=[USE_GPU])).to_csv(os.path.join(args.out,"geodesic_twoscale_params.csv"), index=False)

backend = "GPU (CuPy)" if USE_GPU else "CPU (NumPy)"
print(f"DONE with {backend}. Params: α={a:.3f}, w={w:.3f}, β1={b1:.3f}, β2={b2:.3f}")
