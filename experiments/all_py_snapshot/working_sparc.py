#!/usr/bin/env python3
"""
Diagnostic SPARC Analysis - Figure out why fits are failing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
import glob
import os
from pathlib import Path

# GPU imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("CuPy not available - using CPU")

def read_sparc_file(filepath):
    """Read a single SPARC rotmod file."""
    try:
        data = pd.read_csv(filepath, sep=r'\s+', comment='#',
                          names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
        
        data = data.dropna(subset=['Rad', 'Vobs'])
        data = data[(data['Rad'] > 0) & (data['Vobs'] > 0)]
        
        return data if len(data) >= 3 else None
        
    except Exception:
        return None

def geodesic_model(r, M_central, ell, alpha, stellar_scale, stellar_mass):
    """Geodesic rotation curve model."""
    if GPU_AVAILABLE:
        r = cp.asarray(r)
    
    soi_radius = 0.1
    v_central = cp.where(r < soi_radius,
                        cp.sqrt(M_central / cp.maximum(r, 1e-6)),
                        cp.sqrt(M_central / soi_radius) * cp.sqrt(soi_radius / r))
    
    x = r / (2 * stellar_scale)
    v_stellar_sq = stellar_mass * x**2 * (2.15 * x / (1 + 1.1*x)**2.2)
    v_stellar = cp.sqrt(cp.maximum(v_stellar_sq, 0))
    
    v_newton = cp.sqrt(v_central**2 + v_stellar**2)
    
    weight = cp.exp(-0.5 * (r/ell)**2)
    v_geodesic = v_newton * (1 + alpha * weight)
    
    if GPU_AVAILABLE and hasattr(v_geodesic, 'get'):
        return v_geodesic.get()
    return v_geodesic

def fit_galaxy_extended(radius, velocity, velocity_err=None, verbose=False):
    """
    Fit geodesic model with extended parameter ranges and diagnostics.
    """
    
    def objective(params):
        M_central, ell, alpha, stellar_scale, stellar_mass = params
        
        try:
            v_model = geodesic_model(radius, M_central, ell, alpha, stellar_scale, stellar_mass)
            
            if velocity_err is not None:
                weights = 1.0 / (velocity_err**2 + 0.01**2)
                chi2 = np.sum(weights * (velocity - v_model)**2)
            else:
                chi2 = np.sum((velocity - v_model)**2)
                
            return chi2
            
        except Exception as e:
            if verbose:
                print(f"    Model evaluation failed: {e}")
            return 1e10
    
    # EXTENDED parameter bounds - let's see what the data wants
    extended_bounds = [
        (0.001, 100.0),   # M_central - much wider range
        (0.01, 50.0),     # ell - allow very large geodesic scales
        (0.0, 10.0),      # alpha - allow stronger coupling
        (0.1, 50.0),      # stellar_scale - wider range
        (0.01, 1000.0)    # stellar_mass - much wider range
    ]
    
    # Try multiple initial conditions
    initial_guesses = [
        [0.1, 1.0, 0.3, 2.0, 1.0],     # Original guess
        [1.0, 5.0, 1.0, 5.0, 10.0],    # Larger scale
        [0.01, 10.0, 2.0, 1.0, 0.1],   # Different regime
        [10.0, 0.5, 0.1, 10.0, 100.0], # Yet another regime
    ]
    
    best_result = None
    best_chi2 = np.inf
    
    for i, initial in enumerate(initial_guesses):
        try:
            result = minimize(objective, initial, bounds=extended_bounds, 
                            method='L-BFGS-B', options={'maxiter': 2000})
            
            if result.success and result.fun < best_chi2:
                best_result = result
                best_chi2 = result.fun
                
        except Exception as e:
            if verbose:
                print(f"    Initial guess {i+1} failed: {e}")
            continue
    
    if best_result is not None and best_result.success:
        # Calculate fit statistics
        v_best = geodesic_model(radius, *best_result.x)
        residuals = velocity - v_best
        
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((velocity - np.mean(velocity))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        correlation, p_value = pearsonr(velocity, v_best)
        
        # Check if parameters hit bounds
        bounds_hit = []
        for i, (param, (lower, upper)) in enumerate(zip(best_result.x, extended_bounds)):
            if abs(param - lower) < 0.01 * (upper - lower):
                bounds_hit.append(f"lower_{i}")
            elif abs(param - upper) < 0.01 * (upper - lower):
                bounds_hit.append(f"upper_{i}")
        
        return {
            'success': True,
            'r_squared': r_squared,
            'correlation': correlation,
            'chi2': best_result.fun,
            'bounds_hit': bounds_hit,
            'params': {
                'M_central': best_result.x[0],
                'ell': best_result.x[1],
                'alpha': best_result.x[2],
                'stellar_scale': best_result.x[3],
                'stellar_mass': best_result.x[4]
            },
            'model_curve': v_best
        }
    else:
        return {'success': False, 'message': 'All optimization attempts failed'}

def plot_rotation_curves(galaxies_to_plot, sparc_directory, output_dir='diagnostic_plots'):
    """Plot rotation curves for specific galaxies."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for galaxy_name in galaxies_to_plot:
        filepath = os.path.join(sparc_directory, f"{galaxy_name}_rotmod.dat")
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        data = read_sparc_file(filepath)
        if data is None:
            continue
            
        radius = data['Rad'].values
        velocity = data['Vobs'].values
        velocity_err = data['errV'].values if 'errV' in data.columns else None
        
        print(f"\nDiagnosing {galaxy_name}...")
        print(f"  Data: {len(radius)} points, R = {radius.min():.2f}-{radius.max():.2f} kpc")
        print(f"  Velocity: {velocity.min():.1f}-{velocity.max():.1f} km/s")
        
        # Fit with extended parameters
        fit_result = fit_galaxy_extended(radius, velocity, velocity_err, verbose=True)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Data
        if velocity_err is not None:
            ax1.errorbar(radius, velocity, yerr=velocity_err, fmt='ko', alpha=0.7, label='Observed')
        else:
            ax1.plot(radius, velocity, 'ko', alpha=0.7, label='Observed')
        
        if fit_result['success']:
            ax1.plot(radius, fit_result['model_curve'], 'r-', linewidth=2, label='Geodesic Model')
            
            # Parameters in title
            params = fit_result['params']
            title = f"{galaxy_name}: R² = {fit_result['r_squared']:.3f}\n"
            title += f"ell = {params['ell']:.2f} kpc, α = {params['alpha']:.3f}, "
            title += f"M_c = {params['M_central']:.3f}, M_s = {params['stellar_mass']:.1f}"
            
            if fit_result['bounds_hit']:
                title += f"\nBounds hit: {fit_result['bounds_hit']}"
            
            ax1.set_title(title)
            
            # Residuals
            residuals = velocity - fit_result['model_curve']
            ax2.plot(radius, residuals, 'ko', alpha=0.7)
            ax2.axhline(0, color='r', linestyle='--')
            ax2.set_ylabel('Residuals (km/s)')
            
        else:
            ax1.set_title(f"{galaxy_name}: FIT FAILED")
        
        ax1.set_ylabel('Velocity (km/s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Radius (kpc)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{galaxy_name}_diagnostic.png"), dpi=150)
        plt.close()
        
        # Print diagnostic info
        if fit_result['success']:
            print(f"  FIT: R² = {fit_result['r_squared']:.3f}")
            print(f"  Params: ell = {params['ell']:.2f}, α = {params['alpha']:.3f}")
            if fit_result['bounds_hit']:
                print(f"  WARNING: Hit parameter bounds: {fit_result['bounds_hit']}")
        else:
            print(f"  FAILED: {fit_result.get('message', 'Unknown error')}")

def main():
    print("SPARC Diagnostic Analysis - Extended Parameter Fitting")
    print("=" * 70)
    
    sparc_directory = r"C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts"
    
    # Specific galaxies to diagnose
    # Best fits from previous run
    best_galaxies = ['CamB', 'D564-8', 'UGC00891', 'DDO154', 'DDO170']
    
    # Some of the worst fits
    worst_galaxies = ['NGC0891', 'NGC2955', 'NGC3893', 'NGC5005', 'UGC07690']
    
    # Mix of moderate fits
    moderate_galaxies = ['UGC01281', 'UGC06628', 'UGC07089']
    
    print("Analyzing best fits...")
    plot_rotation_curves(best_galaxies, sparc_directory, 'best_fits')
    
    print("\nAnalyzing worst fits...")
    plot_rotation_curves(worst_galaxies, sparc_directory, 'worst_fits')
    
    print("\nAnalyzing moderate fits...")
    plot_rotation_curves(moderate_galaxies, sparc_directory, 'moderate_fits')
    
    print(f"\nDiagnostic plots saved to:")
    print(f"  best_fits/ - Examples of successful geodesic fits")
    print(f"  worst_fits/ - Examples of failed fits (to understand why)")
    print(f"  moderate_fits/ - Borderline cases")
    
    print(f"\nKey questions to investigate:")
    print(f"1. Are worst fits hitting parameter bounds?")
    print(f"2. Do worst fits have different rotation curve shapes?")
    print(f"3. Can extended parameter ranges fix the failures?")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()