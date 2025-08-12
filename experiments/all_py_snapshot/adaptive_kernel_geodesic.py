#!/usr/bin/env python3
"""
Adaptive Kernel Geodesic Model
Kernel scale adapts to galaxy size - should fix large galaxy problems
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

def adaptive_geodesic_model(r, M_central, ell_fraction, alpha, stellar_scale, stellar_mass, R_galaxy):
    """
    Adaptive geodesic model where kernel scale adapts to galaxy size.
    
    Key insight: ell should scale with the galaxy's characteristic size
    ell_effective = ell_fraction * R_galaxy
    
    This ensures the geodesic coupling operates over the right physical scale
    for each galaxy, regardless of its size.
    """
    
    if GPU_AVAILABLE:
        r = cp.asarray(r)
    
    # ADAPTIVE SCALING: Geodesic length scale proportional to galaxy size
    ell_effective = ell_fraction * R_galaxy
    
    # Ensure reasonable bounds
    ell_effective = cp.clip(ell_effective, 0.1, 100.0)
    
    # Central black hole component (SOI scaling with galaxy mass)
    # Larger galaxies likely have more massive central BHs
    M_central_effective = M_central * (R_galaxy / 10.0)**0.5  # Weak scaling
    soi_radius = 0.1 * (M_central_effective / 1.0)**0.5
    
    v_central = cp.where(r < soi_radius,
                        cp.sqrt(M_central_effective / cp.maximum(r, 1e-6)),
                        cp.sqrt(M_central_effective / soi_radius) * cp.sqrt(soi_radius / r))
    
    # Stellar disk component - scale length can scale with galaxy size
    stellar_scale_effective = stellar_scale * (R_galaxy / 10.0)**0.3  # Mild scaling
    x = r / (2 * stellar_scale_effective)
    v_stellar_sq = stellar_mass * x**2 * (2.15 * x / (1 + 1.1*x)**2.2)
    v_stellar = cp.sqrt(cp.maximum(v_stellar_sq, 0))
    
    # Standard Newtonian combination
    v_newton = cp.sqrt(v_central**2 + v_stellar**2)
    
    # ADAPTIVE GEODESIC KERNEL
    # The key insight: geodesic correlation length should match galaxy scale
    weight = cp.exp(-0.5 * (r/ell_effective)**2)
    geodesic_factor = 1 + alpha * weight
    
    # Apply geodesic enhancement
    v_geodesic = v_newton * geodesic_factor
    
    if GPU_AVAILABLE and hasattr(v_geodesic, 'get'):
        return v_geodesic.get()
    return v_geodesic

def estimate_galaxy_size(radius, velocity):
    """
    Estimate the characteristic size of the galaxy.
    Multiple methods for robustness.
    """
    
    # Method 1: Maximum radius where we have data
    R_max = np.max(radius)
    
    # Method 2: Radius where rotation curve flattens (if it does)
    # Look for where velocity derivative becomes small
    if len(velocity) > 3:
        dv_dr = np.gradient(velocity, radius)
        flat_indices = np.where(np.abs(dv_dr) < 0.1 * np.max(np.abs(dv_dr)))[0]
        if len(flat_indices) > 0:
            R_flat = radius[flat_indices[0]]
        else:
            R_flat = R_max
    else:
        R_flat = R_max
    
    # Method 3: Scale from characteristic velocity (Tully-Fisher-like)
    v_char = np.mean(velocity[-3:]) if len(velocity) >= 3 else np.max(velocity)
    R_tf = v_char / 50.0  # Empirical Tully-Fisher-like relation
    
    # Combine methods - use the most conservative (largest) estimate
    # This ensures we don't underestimate galaxy size
    R_galaxy = max(R_max, R_flat, R_tf)
    
    # Reasonable bounds
    R_galaxy = min(max(R_galaxy, 1.0), 50.0)
    
    return R_galaxy

def fit_adaptive_model(radius, velocity, velocity_err=None, verbose=False):
    """
    Fit the adaptive geodesic model.
    """
    
    # Estimate galaxy size
    R_galaxy = estimate_galaxy_size(radius, velocity)
    
    if verbose:
        print(f"    Estimated galaxy size: R_galaxy = {R_galaxy:.2f} kpc")
        print(f"    Data extends to: {np.max(radius):.2f} kpc")
        print(f"    Ratio: {R_galaxy/np.max(radius):.2f}")
    
    def objective(params):
        M_central, ell_fraction, alpha, stellar_scale, stellar_mass = params
        
        try:
            v_model = adaptive_geodesic_model(
                radius, M_central, ell_fraction, alpha, stellar_scale, stellar_mass, R_galaxy
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
    
    # Parameter bounds
    bounds = [
        (0.001, 100.0),   # M_central
        (0.1, 2.0),       # ell_fraction: geodesic scale as fraction of galaxy size
        (0.0, 10.0),      # alpha: geodesic coupling strength
        (0.1, 10.0),      # stellar_scale 
        (0.01, 1000.0),   # stellar_mass
    ]
    
    # Multiple initial guesses
    initial_guesses = [
        [1.0, 0.3, 0.5, 2.0, 10.0],    # 30% of galaxy size
        [0.1, 0.5, 1.0, 1.0, 1.0],     # 50% of galaxy size
        [10.0, 0.2, 2.0, 5.0, 100.0],  # 20% of galaxy size
        [0.5, 0.8, 0.3, 3.0, 50.0],    # 80% of galaxy size
        [5.0, 1.0, 1.5, 1.5, 200.0],   # Full galaxy size
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
        v_best = adaptive_geodesic_model(radius, *best_result.x, R_galaxy)
        
        residuals = velocity - v_best
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((velocity - np.mean(velocity))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        correlation, p_value = pearsonr(velocity, v_best)
        
        # Calculate effective parameters
        ell_effective = best_result.x[1] * R_galaxy
        
        return {
            'success': True,
            'r_squared': r_squared,
            'correlation': correlation,
            'chi2': best_result.fun,
            'R_galaxy': R_galaxy,
            'params': {
                'M_central': best_result.x[0],
                'ell_fraction': best_result.x[1],
                'alpha': best_result.x[2],
                'stellar_scale': best_result.x[3],
                'stellar_mass': best_result.x[4],
                'ell_effective': ell_effective,
            },
            'model_curve': v_best
        }
    else:
        return {'success': False, 'message': 'All optimization attempts failed'}

def test_adaptive_model():
    """
    Test the adaptive model on problematic large galaxies.
    """
    
    sparc_directory = r"C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts"
    
    # Test galaxies - focus on the size range
    test_galaxies = [
        # Small galaxies (worked before)
        ('CamB', 'Small'),
        ('D564-8', 'Small'),
        ('UGC06628', 'Small'),
        
        # Medium galaxies
        ('UGC01281', 'Medium'),
        ('DDO154', 'Medium'),
        
        # Large galaxies (failed before)
        ('NGC0891', 'Large'),
        ('NGC2955', 'Large'), 
        ('NGC3893', 'Large'),
        ('NGC5005', 'Large'),
    ]
    
    print("ADAPTIVE KERNEL GEODESIC MODEL TEST")
    print("=" * 70)
    print("Testing kernel that adapts to galaxy size...")
    
    results = {}
    
    for galaxy_name, size_class in test_galaxies:
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
        
        print(f"\nAnalyzing {galaxy_name} ({size_class})...")
        print(f"  Data: {len(radius)} points, R = {radius.min():.2f}-{radius.max():.2f} kpc")
        print(f"  Velocity: V = {velocity.min():.1f}-{velocity.max():.1f} km/s")
        
        # Fit adaptive model
        fit_result = fit_adaptive_model(radius, velocity, velocity_err, verbose=True)
        results[galaxy_name] = fit_result
        
        if fit_result['success']:
            r2 = fit_result['r_squared']
            params = fit_result['params']
            R_gal = fit_result['R_galaxy']
            
            print(f"  ✓ SUCCESS: R² = {r2:.3f}")
            print(f"    Galaxy size: R_galaxy = {R_gal:.2f} kpc")
            print(f"    Geodesic scale: ell_effective = {params['ell_effective']:.2f} kpc ({params['ell_fraction']:.1f}% of galaxy)")
            print(f"    Coupling: α = {params['alpha']:.3f}")
        else:
            print(f"  ✗ FAILED: {fit_result.get('message', 'Unknown error')}")
    
    # Summary by size class
    size_classes = {'Small': [], 'Medium': [], 'Large': []}
    
    for (galaxy_name, size_class), result in zip(test_galaxies, results.values()):
        if result['success']:
            size_classes[size_class].append(result['r_squared'])
    
    print(f"\n" + "=" * 70)
    print("ADAPTIVE MODEL RESULTS BY GALAXY SIZE")
    print("=" * 70)
    
    for size_class, r2_values in size_classes.items():
        if len(r2_values) > 0:
            mean_r2 = np.mean(r2_values)
            excellent_fits = sum(1 for r2 in r2_values if r2 > 0.8)
            print(f"{size_class:6s} galaxies: Mean R² = {mean_r2:.3f}, Excellent fits = {excellent_fits}/{len(r2_values)}")
        else:
            print(f"{size_class:6s} galaxies: No successful fits")
    
    # Overall assessment
    all_successful = [r for r in results.values() if r['success']]
    if len(all_successful) > 0:
        all_r2 = [r['r_squared'] for r in all_successful]
        all_excellent = sum(1 for r2 in all_r2 if r2 > 0.8)
        
        print(f"\nOVERALL: {len(all_successful)}/{len(test_galaxies)} successful")
        print(f"Mean R² = {np.mean(all_r2):.3f}")
        print(f"Excellent fits = {all_excellent}/{len(all_successful)} ({all_excellent/len(all_successful)*100:.1f}%)")
        
        if np.mean(all_r2) > 0.6 and all_excellent/len(all_successful) > 0.6:
            print("\n🎯 SUCCESS: Adaptive kernel significantly improves fits!")
        elif np.mean(all_r2) > 0.3:
            print("\n📈 IMPROVEMENT: Adaptive kernel helps but needs refinement")
        else:
            print("\n🤔 MIXED: Some improvement but fundamental issues remain")
    
    return results

if __name__ == "__main__":
    try:
        results = test_adaptive_model()
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()