#!/usr/bin/env python3
"""
Mass-Dependent Geodesic Model - Physically Motivated Scaling
Based on fundamental physics principles, not arbitrary fitting
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

def estimate_galaxy_characteristics(radius, velocity):
    """
    Estimate key galaxy characteristics for scaling laws.
    Based on observational data.
    """
    # Characteristic velocity (roughly flat part of rotation curve)
    v_char = np.mean(velocity[-3:]) if len(velocity) >= 3 else np.max(velocity)
    
    # Characteristic radius (where rotation curve flattens)
    r_char = radius[-1] if len(radius) > 1 else radius[0]
    
    # Estimate total mass using virial theorem: M ~ v²r/G
    # Using G=1 units, typical factor ~1-3 for galaxies
    M_total = 2.0 * v_char**2 * r_char  # Virial mass estimate
    
    # Dynamical time: t_dyn ~ r/v
    t_dyn = r_char / v_char
    
    return {
        'v_char': v_char,
        'r_char': r_char, 
        'M_total': M_total,
        't_dyn': t_dyn
    }

def mass_dependent_geodesic_model(r, M_central, ell_0, alpha_0, stellar_scale, stellar_mass, 
                                 v_char, M_total, mass_scaling_exp, velocity_scaling_exp):
    """
    Mass-dependent geodesic rotation curve model.
    
    Physical scaling laws:
    1. Geodesic length scale: ell ~ (M_total)^mass_scaling_exp
    2. Geodesic coupling: alpha ~ (v_char)^velocity_scaling_exp
    
    Based on:
    - General relativity: stronger fields → different geodesic behavior
    - Virial theorem: characteristic scales set by mass and velocity
    - Galaxy scaling relations: observed correlations
    """
    
    if GPU_AVAILABLE:
        r = cp.asarray(r)
    
    # Reference scales (typical Milky Way-like galaxy)
    M_ref = 1e11  # Reference mass (solar masses, in our units)
    v_ref = 100.0  # Reference velocity (km/s)
    
    # Physics-based scaling laws
    # 1. Geodesic correlation length scales with gravitational radius
    ell_effective = ell_0 * (M_total / M_ref)**mass_scaling_exp
    
    # 2. Geodesic coupling strength scales with field strength/velocity
    alpha_effective = alpha_0 * (v_char / v_ref)**velocity_scaling_exp
    
    # Ensure reasonable bounds (physical constraints)
    ell_effective = cp.clip(ell_effective, 0.1, 50.0)
    alpha_effective = cp.clip(alpha_effective, 0.0, 10.0)
    
    # Central black hole component (SOI scaling)
    soi_radius = 0.1 * (M_central / 1.0)**0.5  # BH sphere of influence scales with mass
    v_central = cp.where(r < soi_radius,
                        cp.sqrt(M_central / cp.maximum(r, 1e-6)),
                        cp.sqrt(M_central / soi_radius) * cp.sqrt(soi_radius / r))
    
    # Stellar disk component
    x = r / (2 * stellar_scale)
    v_stellar_sq = stellar_mass * x**2 * (2.15 * x / (1 + 1.1*x)**2.2)
    v_stellar = cp.sqrt(cp.maximum(v_stellar_sq, 0))
    
    # Standard Newtonian combination
    v_newton = cp.sqrt(v_central**2 + v_stellar**2)
    
    # Mass-dependent geodesic enhancement
    weight = cp.exp(-0.5 * (r/ell_effective)**2)
    geodesic_factor = 1 + alpha_effective * weight
    
    # Apply geodesic enhancement
    v_geodesic = v_newton * geodesic_factor
    
    if GPU_AVAILABLE and hasattr(v_geodesic, 'get'):
        return v_geodesic.get()
    return v_geodesic

def fit_mass_dependent_model(radius, velocity, velocity_err=None, verbose=False):
    """
    Fit the mass-dependent geodesic model.
    """
    
    # Estimate galaxy characteristics
    galaxy_props = estimate_galaxy_characteristics(radius, velocity)
    v_char = galaxy_props['v_char']
    M_total = galaxy_props['M_total']
    
    if verbose:
        print(f"    Galaxy characteristics: v_char = {v_char:.1f} km/s, M_total = {M_total:.2e}")
    
    def objective(params):
        M_central, ell_0, alpha_0, stellar_scale, stellar_mass, mass_exp, vel_exp = params
        
        try:
            v_model = mass_dependent_geodesic_model(
                radius, M_central, ell_0, alpha_0, stellar_scale, stellar_mass,
                v_char, M_total, mass_exp, vel_exp
            )
            
            if velocity_err is not None:
                weights = 1.0 / (velocity_err**2 + 0.01**2)
                chi2 = np.sum(weights * (velocity - v_model)**2)
            else:
                chi2 = np.sum((velocity - v_model)**2)
                
            return chi2
            
        except Exception as e:
            if verbose:
                print(f"      Model evaluation failed: {e}")
            return 1e10
    
    # Parameter bounds - physically motivated
    bounds = [
        (0.001, 100.0),   # M_central: wide range for BH masses
        (0.1, 20.0),      # ell_0: base geodesic length scale 
        (0.0, 5.0),       # alpha_0: base geodesic coupling
        (0.1, 20.0),      # stellar_scale: stellar disk scale
        (0.01, 1000.0),   # stellar_mass: total stellar mass
        (-0.5, 1.0),      # mass_scaling_exp: physically reasonable range
        (-1.0, 1.0),      # velocity_scaling_exp: physically reasonable range
    ]
    
    # Multiple initial guesses
    initial_guesses = [
        [1.0, 2.0, 0.5, 3.0, 10.0, 0.2, 0.0],     # Moderate mass scaling
        [0.1, 1.0, 1.0, 2.0, 1.0, 0.5, -0.2],     # Strong mass scaling, weak velocity scaling
        [10.0, 5.0, 0.3, 5.0, 100.0, 0.0, 0.3],   # No mass scaling, velocity scaling
        [0.5, 3.0, 2.0, 1.0, 50.0, 0.3, -0.5],    # Mixed scaling
    ]
    
    best_result = None
    best_chi2 = np.inf
    
    for i, initial in enumerate(initial_guesses):
        try:
            result = minimize(objective, initial, bounds=bounds, 
                            method='L-BFGS-B', options={'maxiter': 3000})
            
            if result.success and result.fun < best_chi2:
                best_result = result
                best_chi2 = result.fun
                
        except Exception as e:
            if verbose:
                print(f"      Initial guess {i+1} failed: {e}")
            continue
    
    if best_result is not None and best_result.success:
        # Calculate fit statistics
        v_best = mass_dependent_geodesic_model(
            radius, *best_result.x[:5], v_char, M_total, best_result.x[5], best_result.x[6]
        )
        
        residuals = velocity - v_best
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((velocity - np.mean(velocity))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        correlation, p_value = pearsonr(velocity, v_best)
        
        # Calculate effective parameters for this galaxy
        M_ref = 1e11
        v_ref = 100.0
        ell_effective = best_result.x[1] * (M_total / M_ref)**best_result.x[5]
        alpha_effective = best_result.x[2] * (v_char / v_ref)**best_result.x[6]
        
        return {
            'success': True,
            'r_squared': r_squared,
            'correlation': correlation,
            'chi2': best_result.fun,
            'galaxy_props': galaxy_props,
            'params': {
                'M_central': best_result.x[0],
                'ell_0': best_result.x[1],
                'alpha_0': best_result.x[2],
                'stellar_scale': best_result.x[3],
                'stellar_mass': best_result.x[4],
                'mass_scaling_exp': best_result.x[5],
                'velocity_scaling_exp': best_result.x[6],
                'ell_effective': ell_effective,
                'alpha_effective': alpha_effective
            },
            'model_curve': v_best
        }
    else:
        return {'success': False, 'message': 'All optimization attempts failed'}

def analyze_sample_galaxies():
    """
    Test the mass-dependent model on a representative sample.
    """
    
    sparc_directory = r"C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts"
    
    # Test galaxies spanning the mass range
    test_galaxies = [
        # Low-mass/velocity (should work well)
        'CamB', 'D564-8', 'DDO154',
        # Intermediate 
        'UGC01281', 'UGC06628',
        # High-mass/velocity (previously failed)
        'NGC0891', 'NGC2955', 'NGC3893', 'NGC5005'
    ]
    
    print("MASS-DEPENDENT GEODESIC MODEL TEST")
    print("=" * 70)
    print("Testing across galaxy mass/velocity range...")
    
    results = {}
    
    for galaxy_name in test_galaxies:
        filepath = os.path.join(sparc_directory, f"{galaxy_name}_rotmod.dat")
        
        if not os.path.exists(filepath):
            print(f"File not found: {galaxy_name}")
            continue
            
        data = read_sparc_file(filepath)
        if data is None:
            print(f"Could not read data: {galaxy_name}")
            continue
            
        radius = data['Rad'].values
        velocity = data['Vobs'].values
        velocity_err = data['errV'].values if 'errV' in data.columns else None
        
        print(f"\nAnalyzing {galaxy_name}...")
        print(f"  Data: {len(radius)} points, V = {velocity.min():.1f}-{velocity.max():.1f} km/s")
        
        # Fit mass-dependent model
        fit_result = fit_mass_dependent_model(radius, velocity, velocity_err, verbose=True)
        results[galaxy_name] = fit_result
        
        if fit_result['success']:
            r2 = fit_result['r_squared']
            params = fit_result['params']
            props = fit_result['galaxy_props']
            
            print(f"  ✓ SUCCESS: R² = {r2:.3f}")
            print(f"    Galaxy: v_char = {props['v_char']:.1f} km/s, M_total = {props['M_total']:.2e}")
            print(f"    Base params: ell_0 = {params['ell_0']:.2f} kpc, α_0 = {params['alpha_0']:.3f}")
            print(f"    Scaling: mass_exp = {params['mass_scaling_exp']:.3f}, vel_exp = {params['velocity_scaling_exp']:.3f}")
            print(f"    Effective: ell_eff = {params['ell_effective']:.2f} kpc, α_eff = {params['alpha_effective']:.3f}")
        else:
            print(f"  ✗ FAILED: {fit_result.get('message', 'Unknown error')}")
    
    # Summary
    successful_fits = [r for r in results.values() if r['success']]
    
    if len(successful_fits) > 0:
        r2_values = [r['r_squared'] for r in successful_fits]
        mass_exp_values = [r['params']['mass_scaling_exp'] for r in successful_fits]
        vel_exp_values = [r['params']['velocity_scaling_exp'] for r in successful_fits]
        
        print(f"\n" + "=" * 70)
        print("MASS-DEPENDENT MODEL RESULTS")
        print("=" * 70)
        print(f"Successful fits: {len(successful_fits)}/{len(test_galaxies)}")
        print(f"Mean R²: {np.mean(r2_values):.3f} ± {np.std(r2_values):.3f}")
        print(f"Mass scaling exponent: {np.mean(mass_exp_values):.3f} ± {np.std(mass_exp_values):.3f}")
        print(f"Velocity scaling exponent: {np.mean(vel_exp_values):.3f} ± {np.std(vel_exp_values):.3f}")
        
        excellent_fits = sum(1 for r2 in r2_values if r2 > 0.8)
        print(f"Excellent fits (R² > 0.8): {excellent_fits}/{len(successful_fits)} ({excellent_fits/len(successful_fits)*100:.1f}%)")
        
        # Physical interpretation
        avg_mass_exp = np.mean(mass_exp_values)
        avg_vel_exp = np.mean(vel_exp_values)
        
        print(f"\nPHYSICAL INTERPRETATION:")
        if abs(avg_mass_exp) > 0.1:
            if avg_mass_exp > 0:
                print(f"✓ Geodesic length scale INCREASES with galaxy mass (ell ∝ M^{avg_mass_exp:.2f})")
            else:
                print(f"✓ Geodesic length scale DECREASES with galaxy mass (ell ∝ M^{avg_mass_exp:.2f})")
        else:
            print("~ Geodesic length scale is approximately mass-independent")
            
        if abs(avg_vel_exp) > 0.1:
            if avg_vel_exp > 0:
                print(f"✓ Geodesic coupling INCREASES with velocity (α ∝ v^{avg_vel_exp:.2f})")
            else:
                print(f"✓ Geodesic coupling DECREASES with velocity (α ∝ v^{avg_vel_exp:.2f})")
        else:
            print("~ Geodesic coupling is approximately velocity-independent")
        
        print("=" * 70)
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_sample_galaxies()
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()