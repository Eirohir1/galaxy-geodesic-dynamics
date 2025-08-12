import os, glob, time, json, argparse, warnings
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.optimize import minimize_scalar
from scipy import stats

warnings.filterwarnings("ignore")

# -------------------------
# Config: universal choices
# -------------------------
UNIVERSAL_ELL_FACTOR = 0.25   # ell = 0.25 * R_galaxy (fixed)
UNIVERSAL_G_INF      = 0.0    # no offset (fixed)
ALPHA_BOUNDS         = (0.01, 2.0)
REASONABLE_CHI2_MAX  = 1000.0
NULL_TEST_MAX_CHI2   = 100.0  # only do null test for decent fits
DR_CONV              = 0.1    # fixed dr for convolution grid

# -------------------------
# Data model
# -------------------------
@dataclass
class RotmodData:
    r_kpc: np.ndarray
    v_obs: np.ndarray
    dv_obs: np.ndarray
    v_gas: Optional[np.ndarray] = None
    v_disk: Optional[np.ndarray] = None
    v_bulge: Optional[np.ndarray] = None
    name: str = ""

def read_rotmod(path: str) -> RotmodData:
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#;":
                continue
            parts = s.replace(",", " ").split()
            vals = []
            good = True
            for x in parts:
                try:
                    vals.append(float(x))
                except ValueError:
                    good = False
                    break
            if good and len(vals) >= 3:
                rows.append(vals)
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    A = np.array(rows, float)
    if A.shape[1] < 3:
        raise ValueError(f"Need at least 3 columns (r, v_obs, dv) in {path}")
    r = A[:,0]; vobs = A[:,1]; dv = A[:,2]
    vgas  = A[:,3] if A.shape[1] > 3 else None
    vdisk = A[:,4] if A.shape[1] > 4 else None
    vbulg = A[:,5] if A.shape[1] > 5 else None
    name = os.path.basename(path).replace("_rotmod.dat","").replace(".dat","")
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulg, name)

# -------------------------
# Physics-ish pieces
# -------------------------
def v_baryonic(r_kpc: np.ndarray, data: RotmodData) -> np.ndarray:
    v2 = np.zeros_like(r_kpc)
    if data.v_gas  is not None: v2 += np.interp(r_kpc, data.r_kpc, data.v_gas )**2
    if data.v_disk is not None: v2 += np.interp(r_kpc, data.r_kpc, data.v_disk)**2
    if data.v_bulge is not None: v2 += np.interp(r_kpc, data.r_kpc, data.v_bulge)**2
    return np.sqrt(v2)

def model_single_parameter(r_kpc: np.ndarray, data: RotmodData, alpha: float) -> np.ndarray:
    # baryons
    v_bar = v_baryonic(r_kpc, data)
    v_bar_data = v_baryonic(data.r_kpc, data)

    # universal ell
    R = float(np.max(data.r_kpc))
    ell = UNIVERSAL_ELL_FACTOR * R

    # build conv grid
    r_max_conv = max(float(np.max(r_kpc)), R) + 5.0*ell
    r_conv = np.arange(0.0, r_max_conv + DR_CONV, DR_CONV)

    # exponential kernel, properly normalized
    ker = np.exp(-r_conv/ell)
    area = np.trapz(ker, dx=DR_CONV)
    if area <= 0:
        ker[:] = 0.0
    else:
        ker /= area

    # convolve baryonic speed proxy
    v_bar_ext = np.interp(r_conv, data.r_kpc, v_bar_data)
    conv = fftconvolve(v_bar_ext, ker, mode="same") * DR_CONV
    conv_i = np.interp(r_kpc, r_conv, conv)

    v_enh = alpha * conv_i
    v_tot = np.sqrt(np.maximum(v_bar,0.0)**2 + np.maximum(v_enh,0.0)**2)
    return v_tot

def fit_single_parameter(data: RotmodData) -> Tuple[Optional[float], float, Dict]:
    def objective(a: float) -> float:
        if not (ALPHA_BOUNDS[0] <= a <= ALPHA_BOUNDS[1]):
            return 1e10
        try:
            v_pred = model_single_parameter(data.r_kpc, data, a)
            if np.any(~np.isfinite(v_pred)) or np.any(v_pred <= 0):
                return 1e10
            return float(np.sum(((data.v_obs - v_pred)/data.dv_obs)**2))
        except Exception:
            return 1e10

    res = minimize_scalar(objective, bounds=ALPHA_BOUNDS, method="bounded")
    if not res.success:
        return None, 1e10, {"failed": True, "reason": "optimizer_fail"}

    chi2 = float(res.fun)
    if chi2 >= REASONABLE_CHI2_MAX:
        return None, chi2, {"failed": True, "reason": "poor_fit"}

    alpha = float(res.x)
    v_pred = model_single_parameter(data.r_kpc, data, alpha)
    n = len(data.r_kpc)
    red_chi2 = chi2 / max(n-1,1)

    ss_res = float(np.sum((data.v_obs - v_pred)**2))
    ss_tot = float(np.sum((data.v_obs - np.mean(data.v_obs))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

    return alpha, chi2, {
        "reduced_chi2": red_chi2,
        "r_squared": r2,
        "n_points": n,
        "galaxy_size": float(np.max(data.r_kpc)),
        "ell_used": float(UNIVERSAL_ELL_FACTOR*np.max(data.r_kpc)),
        "v_pred": v_pred.tolist()
    }

def null_hypothesis_test(data: RotmodData, your_chi2: float, trials: int = 1000) -> Dict:
    rng = np.random.default_rng(123)
    chi2s = []
    for _ in range(trials):
        a = rng.uniform(*ALPHA_BOUNDS)
        try:
            v = model_single_parameter(data.r_kpc, data, a)
            chi2s.append(float(np.sum(((data.v_obs - v)/data.dv_obs)**2)))
        except Exception:
            chi2s.append(1e10)
    arr = np.array(chi2s, float)
    p = float(np.mean(arr <= your_chi2))
    perc = float(stats.percentileofscore(arr, your_chi2))
    return {
        "your_chi2": your_chi2,
        "random_median_chi2": float(np.median(arr)),
        "p_value": p,
        "percentile": perc,
        "is_significant": p < 0.05,
        "improvement_factor": float(np.median(arr)/max(your_chi2,1e-9))
    }

# -------------------------
# Plotting
# -------------------------
def plot_galaxy(data: RotmodData, alpha: float, metrics: Dict, out_png: str):
    r = data.r_kpc
    v_bar = v_baryonic(r, data)
    v_pred = np.array(metrics["v_pred"])

    plt.figure(figsize=(7,5))
    plt.title(f"{data.name}  (alpha={alpha:.3f}, χ²={metrics['reduced_chi2']*(metrics['n_points']-1):.1f})")
    plt.fill_between(r, data.v_obs - data.dv_obs, data.v_obs + data.dv_obs, alpha=0.2, label="obs ±σ")
    plt.plot(r, data.v_obs, lw=1.8, label="v_obs")
    plt.plot(r, v_bar, lw=1.4, label="v_baryon")
    plt.plot(r, v_pred, lw=2.0, label="model (single-param)")
    plt.xlabel("r [kpc]"); plt.ylabel("v [km/s]")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()

def plot_summaries(success, failed, out_dir):
    if not success:
        return
    alphas = np.array([s["alpha"] for s in success])
    chis  = np.array([s["chi2"] for s in success])
    r2    = np.array([s["metrics"]["r_squared"] for s in success])
    sizes = np.array([s["metrics"]["galaxy_size"] for s in success])

    plt.figure(figsize=(12,9))
    # α distribution
    plt.subplot(2,2,1)
    plt.hist(alphas, bins=25, edgecolor="k", alpha=0.8)
    plt.axvline(np.median(alphas), ls="--", label=f"median={np.median(alphas):.3f}")
    plt.title("α distribution"); plt.legend(); plt.grid(alpha=0.3)

    # χ² (clip to show bulk)
    plt.subplot(2,2,2)
    plt.hist(chis[chis<200], bins=25, edgecolor="k", alpha=0.8)
    plt.title("χ² (clipped at 200)"); plt.grid(alpha=0.3)

    # α vs size
    plt.subplot(2,2,3)
    plt.scatter(sizes, alphas, s=20, alpha=0.7)
    if len(sizes) > 3:
        z = np.polyfit(sizes, alphas, 1); p = np.poly1d(z)
        xx = np.linspace(min(sizes), max(sizes), 100)
        plt.plot(xx, p(xx), "r--", alpha=0.8, label="trend")
        r = np.corrcoef(sizes, alphas)[0,1]
        plt.title(f"α vs galaxy size (r={r:.3f})")
    else:
        plt.title("α vs galaxy size")
    plt.xlabel("max r [kpc]"); plt.ylabel("α"); plt.grid(alpha=0.3); plt.legend()

    # R²
    plt.subplot(2,2,4)
    plt.hist(r2, bins=20, edgecolor="k", alpha=0.8)
    plt.axvline(np.median(r2), ls="--", label=f"median={np.median(r2):.3f}")
    plt.title("R² distribution"); plt.legend(); plt.grid(alpha=0.3)

    plt.suptitle("Single-parameter model — summary", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(os.path.join(out_dir, "summary_plots.png"), dpi=150)
    plt.close()

# -------------------------
# Run all .dat files
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder with .dat files (SPARC/rotmod format).")
    ap.add_argument("--pattern", default="*.dat", help="Filename glob to include (default: *.dat).")
    ap.add_argument("--out", default=None, help="Output folder (default: auto timestamp).")
    ap.add_argument("--null-trials", type=int, default=1000, help="Trials for null test on good fits.")
    args = ap.parse_args()

    data_dir = args.data_dir
    files = sorted(glob.glob(os.path.join(data_dir, args.pattern)))
    if not files:
        print("No .dat files found. Check --data-dir / --pattern.")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(data_dir, f"results_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    per_gal_dir = os.path.join(out_dir, "per_galaxy_plots"); os.makedirs(per_gal_dir, exist_ok=True)

    success: List[Dict] = []
    failed:  List[Dict] = []
    null_tests: List[Dict] = []

    print(f"Scanning {len(files)} files...")
    for i, fp in enumerate(files, 1):
        if i % 25 == 0:
            print(f"  {i}/{len(files)}")
        try:
            data = read_rotmod(fp)
            alpha, chi2, metrics = fit_single_parameter(data)
            if alpha is None:
                failed.append({
                    "name": data.name, "file": fp, "reason": metrics.get("reason","unknown"),
                    "chi2_attempted": chi2, "n_points": len(data.r_kpc),
                    "peak_velocity": float(np.max(data.v_obs)), "galaxy_size": float(np.max(data.r_kpc))
                })
                continue

            rec = {"name": data.name, "file": fp, "alpha": alpha, "chi2": chi2, "metrics": metrics}
            success.append(rec)

            # null test for good fits
            if chi2 < NULL_TEST_MAX_CHI2:
                nt = null_hypothesis_test(data, chi2, trials=args.null_trials)
                rec["null_test"] = nt
                null_tests.append(nt)

            # save per-galaxy plot
            out_png = os.path.join(per_gal_dir, f"{data.name}.png")
            plot_galaxy(data, alpha, metrics, out_png)

        except Exception as e:
            failed.append({"name": os.path.basename(fp), "file": fp, "error": str(e)})

    # summary + outputs
    nS, nF = len(success), len(failed)
    success_rate = nS / max(nS + nF, 1)

    results = {
        "universal": {
            "ell_factor": UNIVERSAL_ELL_FACTOR,
            "g_infinity": UNIVERSAL_G_INF,
            "alpha_bounds": ALPHA_BOUNDS
        },
        "counts": {"successful": nS, "failed": nF, "success_rate": success_rate},
        "successful": success,
        "failed": failed,
        "null_tests": null_tests
    }

    # basic stats over successes
    if success:
        alphas = [s["alpha"] for s in success]
        chi2s  = [s["chi2"] for s in success]
        sig = [nt for nt in null_tests if nt.get("is_significant")]
        results["statistics"] = {
            "alpha_median": float(np.median(alphas)),
            "alpha_std": float(np.std(alphas)),
            "alpha_minmax": (float(np.min(alphas)), float(np.max(alphas))),
            "chi2_median": float(np.median(chi2s)),
            "significance_rate": (len(sig)/len(null_tests)) if null_tests else 0.0
        }

    # write JSON + CSV
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # small CSV
    try:
        import csv
        csv_path = os.path.join(out_dir, "results.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name","alpha","chi2","reduced_chi2","r_squared","n_points","galaxy_size","ell_used"])
            for s in success:
                m = s["metrics"]
                w.writerow([s["name"], f"{s['alpha']:.6f}", f"{s['chi2']:.3f}",
                            f"{m['reduced_chi2']:.3f}", f"{m['r_squared']:.3f}",
                            m["n_points"], f"{m['galaxy_size']:.3f}", f"{m['ell_used']:.3f}"])
    except Exception:
        pass

    # plots
    plot_summaries(success, failed, out_dir)

    print("\n=== RUN COMPLETE ===")
    print(f"Files processed: {nS+nF}  |  Success: {nS}  |  Failed: {nF}  |  Rate: {success_rate:.1%}")
    print(f"Output folder: {out_dir}")
    if "statistics" in results:
        st = results["statistics"]
        print(f"Median α: {st['alpha_median']:.3f} ± {st['alpha_std']:.3f}")
        print(f"Median χ²: {st['chi2_median']:.1f}")
        print(f"Significance (null test): {st['significance_rate']:.1%}")

if __name__ == "__main__":
    main()
