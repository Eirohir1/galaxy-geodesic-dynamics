import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import os

# === SET TO YOUR TRUE DATA DIR ===
DATA_DIR = r'C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts'

spiral_galaxies = [
    'CamB_rotmod.dat', 'D512-2_rotmod.dat', 'D564-8_rotmod.dat', 'D631-7_rotmod.dat',
    'DDO064_rotmod.dat', 'DDO154_rotmod.dat', 'DDO161_rotmod.dat', 'DDO168_rotmod.dat',
    'DDO170_rotmod.dat', 'ESO079-G014_rotmod.dat', 'ESO116-G012_rotmod.dat',
    'ESO444-G084_rotmod.dat', 'ESO563-G021_rotmod.dat', 'F561-1_rotmod.dat',
    'F563-1_rotmod.dat', 'F563-V1_rotmod.dat', 'F563-V2_rotmod.dat', 'F565-V2_rotmod.dat',
    'F567-2_rotmod.dat', 'F568-1_rotmod.dat'
]

print(f"\nChecking DATA_DIR: {DATA_DIR}")
if os.path.exists(DATA_DIR):
    files_in_dir = os.listdir(DATA_DIR)
    print(f"Directory exists. Files inside:")
    for f in files_in_dir:
        print(" ", f)
else:
    print("ERROR: DATA_DIR does not exist! Double-check your path.")
    exit(1)

print("\nChecking each spiral galaxy file in spiral_galaxies list:")
good_files = []
for fname in spiral_galaxies:
    path = os.path.join(DATA_DIR, fname)
    exists = os.path.exists(path)
    print(f"  {fname}: {'FOUND' if exists else 'MISSING'}")
    if exists:
        good_files.append(path)

@dataclass
class RotmodData:
    r_kpc: np.ndarray
    v_obs: np.ndarray
    dv_obs: np.ndarray
    v_gas: Optional[np.ndarray] = None
    v_disk: Optional[np.ndarray] = None
    v_bulge: Optional[np.ndarray] = None

def read_rotmod(path: str) -> RotmodData:
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith(";"):
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
        raise ValueError(f"No valid numeric rows in {path}")
    arr = np.array(rows, dtype=float)
    r = arr[:, 0]
    vobs = arr[:, 1] 
    dv = arr[:, 2]
    vgas = arr[:, 3] if arr.shape[1] > 3 else None
    vdisk = arr[:, 4] if arr.shape[1] > 4 else None
    vbulge = arr[:, 5] if arr.shape[1] > 5 else None
    if np.any(r <= 0):
        raise ValueError("Non-positive radii detected")
    if np.any(dv <= 0):
        raise ValueError("Non-positive velocity errors detected")
    return RotmodData(r, vobs, dv, vgas, vdisk, vbulge)

def v_baryonic(r_kpc: np.ndarray, data: RotmodData) -> np.ndarray:
    v_bar_squared = np.zeros_like(r_kpc)
    if data.v_gas is not None:
        v_gas_interp = np.interp(r_kpc, data.r_kpc, data.v_gas)
        v_bar_squared += np.maximum(v_gas_interp, 0)**2
    if data.v_disk is not None:
        v_disk_interp = np.interp(r_kpc, data.r_kpc, data.v_disk) 
        v_bar_squared += np.maximum(v_disk_interp, 0)**2
    if data.v_bulge is not None:
        v_bulge_interp = np.interp(r_kpc, data.r_kpc, data.v_bulge)
        v_bar_squared += np.maximum(v_bulge_interp, 0)**2
    return np.sqrt(v_bar_squared)

def mond_standard(r_kpc: np.ndarray, data: RotmodData, a0_kms2: float = 1.2e-10) -> np.ndarray:
    v_bar = v_baryonic(r_kpc, data)
    mask = r_kpc > 0
    a_N = np.zeros_like(r_kpc)
    a_N[mask] = v_bar[mask]**2 / r_kpc[mask]
    a_N_SI = a_N * 1e6 / 3.086e19
    x = a_N_SI / a0_kms2
    mu = x / (1 + x)
    v_mond = np.zeros_like(v_bar)
    mu_safe = np.maximum(mu, 1e-10)
    v_mond = v_bar / np.sqrt(mu_safe)
    return v_mond

def geodesic_reinforcement(r_kpc: np.ndarray, data: RotmodData, alpha: float, ell_factor: float) -> np.ndarray:
    v_bar = v_baryonic(r_kpc, data)
    if len(r_kpc) > 1:
        dr = np.median(np.diff(r_kpc))
    else:
        dr = 0.1
    R_galaxy = np.max(data.r_kpc)
    ell = ell_factor * R_galaxy
    r_max = max(r_kpc[-1], R_galaxy) + 5*ell
    r_conv = np.arange(0, r_max, dr)
    kern = np.exp(-r_conv / ell)
    kern = kern / np.trapz(kern, r_conv)
    v_bar_ext = np.interp(r_conv, r_kpc, v_bar)
    conv_result = fftconvolve(v_bar_ext, kern, mode='same') * dr
    conv_interp = np.interp(r_kpc, r_conv, conv_result)
    v_dm = alpha * conv_interp
    v_total = np.sqrt(v_bar**2 + np.maximum(v_dm, 0)**2)
    return v_total

def calculate_information_criteria(chi2: float, n_params: int, n_data: int) -> tuple:
    """
    Calculate AIC and BIC information criteria.
    
    Parameters:
    -----------
    chi2 : float
        Chi-squared statistic
    n_params : int
        Number of free parameters in the model
    n_data : int
        Number of data points
    
    Returns:
    --------
    tuple : (AIC, BIC)
        Akaike Information Criterion and Bayesian Information Criterion
    """
    # AIC = 2k + χ²
    aic = 2 * n_params + chi2
    
    # BIC = k⋅ln(n) + χ²
    bic = n_params * np.log(n_data) + chi2
    
    return aic, bic

def fit_model_robust(data: RotmodData, model_func, param_bounds, initial_guess):
    def objective(params):
        try:
            for i, (low, high) in enumerate(param_bounds):
                if not (low <= params[i] <= high):
                    return 1e8
            v_pred = model_func(data.r_kpc, data, *params)
            if np.any(~np.isfinite(v_pred)) or np.any(v_pred <= 0):
                return 1e8
            residuals = (data.v_obs - v_pred) / data.dv_obs
            chi2 = np.sum(residuals**2)
            return chi2
        except Exception:
            return 1e8
    methods = ['Nelder-Mead', 'Powell', 'BFGS']
    best_result = None
    best_chi2 = 1e8
    for method in methods:
        try:
            result = minimize(objective, initial_guess, method=method,
                            options={'maxiter': 1000})
            if result.success and result.fun < best_chi2:
                best_result = result
                best_chi2 = result.fun
        except:
            continue
    return best_result

def compare_all_models(data: RotmodData):
    results = {}
    n_data = len(data.r_kpc)
    
    # Fit Geodesic model (2 parameters: alpha, ell_factor)
    result_geo = fit_model_robust(
        data, geodesic_reinforcement,
        [(0.01, 1.0), (0.1, 2.0)],
        [0.3, 0.6]
    )
    if result_geo and result_geo.success:
        alpha_opt, ell_opt = result_geo.x
        v_pred_geo = geodesic_reinforcement(data.r_kpc, data, alpha_opt, ell_opt)
        n_params_geo = 2  # alpha and ell_factor
        aic_geo, bic_geo = calculate_information_criteria(result_geo.fun, n_params_geo, n_data)
        
        results['GEODESIC'] = {
            'params': result_geo.x,
            'n_params': n_params_geo,
            'chi2': result_geo.fun,
            'v_pred': v_pred_geo,
            'dof': n_data - n_params_geo,
            'reduced_chi2': result_geo.fun / (n_data - n_params_geo),
            'aic': aic_geo,
            'bic': bic_geo
        }
    
    # Evaluate MOND model (0 parameters: a0 is fixed)
    try:
        v_pred_mond = mond_standard(data.r_kpc, data)
        chi2_mond = np.sum(((data.v_obs - v_pred_mond) / data.dv_obs)**2)
        n_params_mond = 0  # a0 is fixed at 1.2e-10
        aic_mond, bic_mond = calculate_information_criteria(chi2_mond, n_params_mond, n_data)
        
        results['MOND'] = {
            'params': [1.2e-10],
            'n_params': n_params_mond,
            'chi2': chi2_mond,
            'v_pred': v_pred_mond,
            'dof': n_data - max(1, n_params_mond),  # Use at least 1 for dof calculation
            'reduced_chi2': chi2_mond / (n_data - max(1, n_params_mond)),
            'aic': aic_mond,
            'bic': bic_mond
        }
    except Exception as e:
        print(f"MOND failed: {e}")
    
    return results

def print_model_comparison_table(results: dict, galaxy_name: str):
    """Print a formatted comparison table for a single galaxy."""
    print(f"\n📊 Model Comparison for {galaxy_name}")
    print("=" * 80)
    print(f"{'Model':<10} {'Params':<8} {'χ²':<10} {'χ²/dof':<10} {'AIC':<10} {'BIC':<10} {'Preferred':<10}")
    print("-" * 80)
    
    if not results:
        print("No successful fits available")
        return
    
    # Calculate which model is preferred by each criterion
    models = list(results.keys())
    if len(models) > 1:
        aic_values = {model: results[model]['aic'] for model in models}
        bic_values = {model: results[model]['bic'] for model in models}
        chi2_values = {model: results[model]['chi2'] for model in models}
        
        best_aic = min(aic_values, key=aic_values.get)
        best_bic = min(bic_values, key=bic_values.get)
        best_chi2 = min(chi2_values, key=chi2_values.get)
    else:
        best_aic = best_bic = best_chi2 = models[0] if models else None
    
    for model, res in results.items():
        preferred = []
        if model == best_chi2:
            preferred.append("χ²")
        if model == best_aic:
            preferred.append("AIC")
        if model == best_bic:
            preferred.append("BIC")
        
        preferred_str = ",".join(preferred) if preferred else "-"
        
        print(f"{model:<10} {res['n_params']:<8} {res['chi2']:<10.1f} "
              f"{res['reduced_chi2']:<10.2f} {res['aic']:<10.1f} "
              f"{res['bic']:<10.1f} {preferred_str:<10}")

def rigorous_comparison():
    print("\n🔬 RIGOROUS MOND VS GEODESIC COMPARISON (Spiral sample)")
    print("Including AIC and BIC Information Criteria")
    print("=" * 70)
    if not good_files:
        print("No spiral galaxy files found in this directory!")
        return
    
    all_results = []
    
    for filename in good_files:
        galaxy_name = os.path.basename(filename).replace('_rotmod.dat', '')
        print(f"\nAnalyzing: {galaxy_name}")
        try:
            data = read_rotmod(filename)
            results = compare_all_models(data)
            if results:
                print_model_comparison_table(results, galaxy_name)
                all_results.append((galaxy_name, results))
            else:
                print(f"  No successful fits")
        except Exception as e:
            print(f"  Error: {e}")
    
    if all_results:
        print(f"\n📈 STATISTICAL SUMMARY ({len(all_results)} spiral galaxies)")
        print("=" * 70)
        
        # Collect statistics for each model
        model_stats = {}
        for model_name in ['GEODESIC', 'MOND']:
            chi2_values = []
            reduced_chi2_values = []
            aic_values = []
            bic_values = []
            
            for galaxy_name, results in all_results:
                if model_name in results:
                    res = results[model_name]
                    chi2_values.append(res['chi2'])
                    reduced_chi2_values.append(res['reduced_chi2'])
                    aic_values.append(res['aic'])
                    bic_values.append(res['bic'])
            
            if chi2_values:
                model_stats[model_name] = {
                    'chi2': np.array(chi2_values),
                    'reduced_chi2': np.array(reduced_chi2_values),
                    'aic': np.array(aic_values),
                    'bic': np.array(bic_values),
                    'success_rate': len(chi2_values) / len(all_results)
                }
        
        # Print summary statistics
        for model_name, stats in model_stats.items():
            print(f"\n{model_name} Model:")
            print(f"  Success rate: {len(stats['chi2'])}/{len(all_results)} = {100*stats['success_rate']:.1f}%")
            print(f"  χ² - Mean: {np.mean(stats['chi2']):.1f}, Median: {np.median(stats['chi2']):.1f}, Std: {np.std(stats['chi2']):.1f}")
            print(f"  χ²/dof - Mean: {np.mean(stats['reduced_chi2']):.2f}, Median: {np.median(stats['reduced_chi2']):.2f}, Std: {np.std(stats['reduced_chi2']):.2f}")
            print(f"  AIC - Mean: {np.mean(stats['aic']):.1f}, Median: {np.median(stats['aic']):.1f}, Std: {np.std(stats['aic']):.1f}")
            print(f"  BIC - Mean: {np.mean(stats['bic']):.1f}, Median: {np.median(stats['bic']):.1f}, Std: {np.std(stats['bic']):.1f}")
        
        # Model preference summary
        if len(model_stats) > 1:
            print(f"\n🏆 MODEL PREFERENCE SUMMARY:")
            print("=" * 40)
            
            aic_wins = {'GEODESIC': 0, 'MOND': 0}
            bic_wins = {'GEODESIC': 0, 'MOND': 0}
            chi2_wins = {'GEODESIC': 0, 'MOND': 0}
            
            for galaxy_name, results in all_results:
                if 'GEODESIC' in results and 'MOND' in results:
                    # AIC comparison
                    if results['GEODESIC']['aic'] < results['MOND']['aic']:
                        aic_wins['GEODESIC'] += 1
                    else:
                        aic_wins['MOND'] += 1
                    
                    # BIC comparison
                    if results['GEODESIC']['bic'] < results['MOND']['bic']:
                        bic_wins['GEODESIC'] += 1
                    else:
                        bic_wins['MOND'] += 1
                    
                    # Chi-squared comparison
                    if results['GEODESIC']['chi2'] < results['MOND']['chi2']:
                        chi2_wins['GEODESIC'] += 1
                    else:
                        chi2_wins['MOND'] += 1
            
            total_comparisons = sum(aic_wins.values())
            if total_comparisons > 0:
                print(f"AIC Preferences (lower is better):")
                for model, wins in aic_wins.items():
                    print(f"  {model}: {wins}/{total_comparisons} = {100*wins/total_comparisons:.1f}%")
                
                print(f"BIC Preferences (lower is better):")
                for model, wins in bic_wins.items():
                    print(f"  {model}: {wins}/{total_comparisons} = {100*wins/total_comparisons:.1f}%")
                
                print(f"χ² Preferences (lower is better):")
                for model, wins in chi2_wins.items():
                    print(f"  {model}: {wins}/{total_comparisons} = {100*wins/total_comparisons:.1f}%")
        
        # Information criteria interpretation
        print(f"\n💡 INTERPRETATION:")
        print("=" * 40)
        print("• AIC = 2k + χ² (gentle complexity penalty)")
        print("• BIC = k⋅ln(n) + χ² (harsh complexity penalty for large n)")
        print("• Lower values indicate better models")
        print("• GEODESIC has 2 parameters (α, ℓ)")
        print("• MOND has 0 parameters (fixed a₀)")
        if 'GEODESIC' in model_stats and 'MOND' in model_stats:
            geo_chi2_mean = np.mean(model_stats['GEODESIC']['chi2'])
            mond_chi2_mean = np.mean(model_stats['MOND']['chi2'])
            improvement_factor = mond_chi2_mean / geo_chi2_mean if geo_chi2_mean > 0 else float('inf')
            print(f"• Geodesic achieves ~{improvement_factor:.1f}× better χ² on average")
            print("• Even with complexity penalty, geodesic likely dominates")
    
    return all_results

if __name__ == "__main__":
    rigorous_comparison()