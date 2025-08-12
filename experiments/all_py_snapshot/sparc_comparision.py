import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
from pathlib import Path

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled")
except ImportError:
    try:
        import jax.numpy as jnp
        from jax import jit, vmap
        from jax.scipy.optimize import minimize as jax_minimize
        GPU_AVAILABLE = True
        print("JAX detected - GPU acceleration enabled")
        cp = jnp  # Use JAX as fallback
    except ImportError:
        import numpy as cp
        GPU_AVAILABLE = False
        print("No GPU libraries found - using CPU")

from scipy.optimize import minimize
from scipy.stats import pearsonr

# ========================
# SPARC Data Reading Functions
# ========================

def read_sparc_rotmod_file(filepath):
    """
    Read a SPARC rotmod .dat file.
    Expected columns: Rad, Vobs, errV, Vgas, Vdisk, Vbul, SBdisk, SBbul
    """
    try:
        # Try reading with standard SPARC format
        data = pd.read_csv(filepath, delim_whitespace=True, 
                          names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'],
                          comment='#')
        
        # Handle potential missing columns
        required_cols = ['Rad', 'Vobs', 'errV']
        for col in required_cols:
            if col not in data.columns or data[col].isna().all():
                print(f"Warning: Missing or empty column {col} in {filepath}")
                return None
                
        # Remove any rows with NaN in critical columns
        data = data.dropna(subset=['Rad', 'Vobs'])
        
        # Ensure positive radii and velocities
        data = data[(data['Rad'] > 0) & (data['Vobs'] > 0)]
        
        if len(data) < 3:  # Need at least 3 points for fitting
            print(f"Warning: Insufficient data points in {filepath}")
            return None
            
        return data
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def load_sparc_database(data_directory):
    """
    Load all SPARC rotmod files from a directory.
    Returns dictionary with galaxy names as keys.
    """
    sparc_data = {}
    
    # Look for .dat files
    dat_files = glob.glob(os.path.join(data_directory, "*.dat"))
    
    if not dat_files:
        print(f"No .dat files found in {data_directory}")
        return sparc_data
    
    print(f"Found {len(dat_files)} potential SPARC files")
    
    for filepath in dat_files:
        # Extract galaxy name from filename
        galaxy_name = Path(filepath).stem.replace('_rotmod', '')
        
        data = read_sparc_rotmod_file(filepath)
        if data is not None:
            sparc_data[galaxy_name] = data
            print(f"Loaded {galaxy_name}: {len(data)} data points")
        
    print(f"Successfully loaded {len(sparc_data)} galaxies")
    return sparc_data

# ========================
# Geodesic Model Implementation
# ========================

def geodesic_rotation_curve_gpu(r, M_central, ell, alpha, stellar_scale, stellar_mass):
    """
    GPU-accelerated geodesic rotation curve calculation.
    """
    # Convert to GPU arrays if available
    if GPU_AVAILABLE:
        r = cp.asarray(r)
    
    # Central black hole component (limited to SOI ~ 0.1 kpc)
    soi_radius = 0.1
    
    # Vectorized conditional for SOI
    mask_inside_soi = r < soi_radius
    v_central_inside = cp.sqrt(M_central / cp.maximum(r, 1e-6))
    v_central_outside = cp.sqrt(M_central / soi_radius) * cp.sqrt(soi_radius / r)
    v_central = cp.where(mask_inside_soi, v_central_inside, v_central_outside)
    
    # Stellar disk component (exponential profile) - GPU vectorized
    x = r / (2 * stellar_scale)
    # Approximation for exponential disk rotation curve
    v_stellar_squared = stellar_mass * x**2 * (2.15 * x / (1 + 1.1*x)**2.2)
    v_stellar = cp.sqrt(cp.maximum(v_stellar_squared, 0))
    
    # Standard Newtonian combination
    v_newton = cp.sqrt(v_central**2 + v_stellar**2)
    
    # Geodesic enhancement - vectorized
    weight = cp.exp(-0.5 * (r/ell)**2)
    geodesic_factor = 1 + alpha * weight
    
    # Apply geodesic enhancement
    v_geodesic = v_newton * geodesic_factor
    
    # Convert back to CPU if needed for compatibility
    if GPU_AVAILABLE and hasattr(v_geodesic, 'get'):
        return v_geodesic.get()
    return v_geodesic

def batch_fit_galaxies_gpu(galaxy_data_list, batch_size=32):
    """
    GPU-accelerated batch fitting of multiple galaxies simultaneously.
    """
    results = {}
    n_galaxies = len(galaxy_data_list)
    
    print(f"GPU batch fitting {n_galaxies} galaxies (batch size: {batch_size})")
    
    for i in range(0, n_galaxies, batch_size):
        batch_end = min(i + batch_size, n_galaxies)
        batch = galaxy_data_list[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}/{(n_galaxies-1)//batch_size + 1}")
        
        # Process batch
        for galaxy_name, data in batch:
            radius = data['Rad'].values
            velocity = data['Vobs'].values
            velocity_err = data['errV'].values if 'errV' in data.columns else None
            
            # Single galaxy fit with GPU acceleration
            fit_result = fit_geodesic_model_gpu(radius, velocity, velocity_err)
            results[galaxy_name] = fit_result
            
            if fit_result['success']:
                print(f"  {galaxy_name}: R² = {fit_result['r_squared']:.3f}")
            else:
                print(f"  {galaxy_name}: FAILED")
    
    return results

def fit_geodesic_model_gpu(radius, velocity, velocity_err=None):
    """
    GPU-accelerated fitting with optimized objective function.
    """
    # Convert to GPU arrays
    if GPU_AVAILABLE:
        radius_gpu = cp.asarray(radius, dtype=cp.float32)
        velocity_gpu = cp.asarray(velocity, dtype=cp.float32)
        if velocity_err is not None:
            velocity_err_gpu = cp.asarray(velocity_err, dtype=cp.float32)
        else:
            velocity_err_gpu = None
    else:
        radius_gpu = radius
        velocity_gpu = velocity
        velocity_err_gpu = velocity_err
    
    # Parameter bounds and initial guess
    initial_params = [0.1, 1.0, 0.3, 2.0, 1.0]  # M_central, ell, alpha, stellar_scale, stellar_mass
    bounds = [(0.01, 10.0), (0.1, 10.0), (0.0, 2.0), (0.5, 10.0), (0.1, 100.0)]
    
    def objective_gpu(params):
        """GPU-optimized objective function."""
        M_central, ell, alpha, stellar_scale, stellar_mass = params
        
        # Calculate model on GPU
        v_model = geodesic_rotation_curve_gpu(radius_gpu, M_central, ell, alpha, 
                                            stellar_scale, stellar_mass)
        
        if GPU_AVAILABLE:
            v_model = cp.asarray(v_model)
        
        # Chi-squared calculation on GPU
        residuals = velocity_gpu - v_model
        
        if velocity_err_gpu is not None:
            weights = 1.0 / (velocity_err_gpu**2 + 0.01**2)
            chi2 = cp.sum(weights * residuals**2)
        else:
            chi2 = cp.sum(residuals**2)
        
        # Convert back to CPU scalar for scipy
        if GPU_AVAILABLE and hasattr(chi2, 'get'):
            return float(chi2.get())
        return float(chi2)
    
    # Perform optimization
    try:
        result = minimize(objective_gpu, initial_params, bounds=bounds, 
                         method='L-BFGS-B', options={'maxiter': 1000})
        
        if result.success:
            # Calculate final statistics on GPU
            v_best = geodesic_rotation_curve_gpu(radius_gpu, *result.x)
            
            if GPU_AVAILABLE:
                v_best_cpu = v_best.get() if hasattr(v_best, 'get') else v_best
                velocity_cpu = velocity_gpu.get() if hasattr(velocity_gpu, 'get') else velocity_gpu
            else:
                v_best_cpu = v_best
                velocity_cpu = velocity_gpu
            
            # Calculate fit statistics
            residuals = velocity_cpu - v_best_cpu
            chi2 = result.fun
            reduced_chi2 = chi2 / (len(velocity) - 5)  # 5 parameters
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((velocity_cpu - np.mean(velocity_cpu))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Correlation coefficient
            correlation, p_value = pearsonr(velocity_cpu, v_best_cpu)
            
            best_params = {
                'M_central': result.x[0],
                'ell': result.x[1], 
                'alpha': result.x[2],
                'stellar_scale': result.x[3],
                'stellar_mass': result.x[4]
            }
            
            return {
                'success': True,
                'params': best_params,
                'chi2': chi2,
                'reduced_chi2': reduced_chi2,
                'r_squared': r_squared,
                'correlation': correlation,
                'p_value': p_value,
                'model_curve': v_best_cpu
            }
        else:
            return {'success': False, 'message': result.message}
            
    except Exception as e:
        return {'success': False, 'message': f"Optimization failed: {str(e)}"}

def analyze_sparc_database_gpu(sparc_data, output_dir='sparc_analysis'):
    """
    GPU-accelerated systematic analysis of the entire SPARC database.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting GPU-accelerated SPARC analysis...")
    
    # Convert to list for batch processing
    galaxy_list = list(sparc_data.items())
    
    # Use batch processing for GPU efficiency
    if GPU_AVAILABLE and len(galaxy_list) > 10:
        results = batch_fit_galaxies_gpu(galaxy_list, batch_size=16)
    else:
        # Fallback to sequential processing
        results = {}
        for i, (galaxy_name, data) in enumerate(galaxy_list):
            print(f"Analyzing {galaxy_name} ({i+1}/{len(galaxy_list)})...")
            
            radius = data['Rad'].values
            velocity = data['Vobs'].values
            velocity_err = data['errV'].values if 'errV' in data.columns else None
            
            fit_result = fit_geodesic_model_gpu(radius, velocity, velocity_err)
            results[galaxy_name] = fit_result
    
    successful_fits = sum(1 for r in results.values() if r['success'])
    print(f"\nGPU analysis complete: {successful_fits}/{len(sparc_data)} successful fits")
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(results)
    
    # Print detailed summary
    print_detailed_summary(summary_stats, results)
    
    # Save results
    save_analysis_results(results, summary_stats, output_dir)
    
    return results, summary_stats

def plot_galaxy_fit(galaxy_name, data, fit_result, save_path=None):
    """
    Plot individual galaxy rotation curve with geodesic model fit.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    radius = data['Rad'].values
    velocity = data['Vobs'].values
    velocity_err = data['errV'].values if 'errV' in data.columns else None
    
    # Top panel: Data and fit
    if velocity_err is not None:
        ax1.errorbar(radius, velocity, yerr=velocity_err, fmt='ko', 
                    alpha=0.7, label='Observed')
    else:
        ax1.plot(radius, velocity, 'ko', alpha=0.7, label='Observed')
    
    if fit_result['success']:
        ax1.plot(radius, fit_result['model_curve'], 'r-', linewidth=2, 
                label='Geodesic Model')
        
        # Add fit statistics to legend
        r2 = fit_result['r_squared']
        corr = fit_result['correlation']
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}\\nCorr = {corr:.3f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_ylabel('Velocity (km/s)')
    ax1.set_title(f'{galaxy_name} - Geodesic Model Fit')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Residuals
    if fit_result['success']:
        residuals = velocity - fit_result['model_curve']
        ax2.plot(radius, residuals, 'ko', alpha=0.7)
        ax2.axhline(0, color='r', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Residuals (km/s)')
        ax2.set_xlabel('Radius (kpc)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_summary_statistics(results):
    """
    Generate summary statistics across all galaxies.
    """
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if not successful_results:
        return {"error": "No successful fits to analyze"}
    
    # Extract fit statistics
    r_squared_values = [r['r_squared'] for r in successful_results.values()]
    correlations = [r['correlation'] for r in successful_results.values()]
    reduced_chi2_values = [r['reduced_chi2'] for r in successful_results.values()]
    
    # Extract parameters
    param_stats = {}
    param_names = ['M_central', 'ell', 'alpha', 'stellar_scale', 'stellar_mass']
    
    for param in param_names:
        values = [r['params'][param] for r in successful_results.values()]
        param_stats[param] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    summary = {
        'n_successful_fits': len(successful_results),
        'n_total_galaxies': len(results),
        'success_rate': len(successful_results) / len(results),
        'fit_quality': {
            'mean_r_squared': np.mean(r_squared_values),
            'median_r_squared': np.median(r_squared_values),
            'mean_correlation': np.mean(correlations),
            'median_correlation': np.median(correlations),
            'mean_reduced_chi2': np.mean(reduced_chi2_values)
        },
        'parameter_statistics': param_stats
    }
    
    return summary

def save_analysis_results(results, summary_stats, output_dir):
    """
    Save analysis results to files.
    """
    # Save summary statistics
    import json
    with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save detailed results as CSV
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        rows = []
        for galaxy, result in successful_results.items():
            row = {'galaxy': galaxy}
            row.update(result['params'])
            row.update({
                'r_squared': result['r_squared'],
                'correlation': result['correlation'],
                'reduced_chi2': result['reduced_chi2']
            })
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, 'fit_results.csv'), index=False)
        
        print(f"Results saved to {output_dir}/")
        print(f"Summary: {summary_stats['fit_quality']['mean_r_squared']:.3f} mean R²")
        print(f"Best geodesic length scale: {summary_stats['parameter_statistics']['ell']['median']:.2f} ± {summary_stats['parameter_statistics']['ell']['std']:.2f} kpc")

# ========================
# Main Analysis Script
# ========================

def print_detailed_summary(summary_stats, results):
    """
    Generate a comprehensive text summary for sharing.
    """
    print("\n" + "=" * 70)
    print("GEODESIC THEORY vs SPARC DATABASE - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    print(f"\nDATA OVERVIEW:")
    print(f"Total galaxies analyzed: {summary_stats['n_total_galaxies']}")
    print(f"Successful fits: {summary_stats['n_successful_fits']}")
    print(f"Success rate: {summary_stats['success_rate']:.1%}")
    
    print(f"\nMODEL PERFORMANCE:")
    fq = summary_stats['fit_quality']
    print(f"Mean R-squared: {fq['mean_r_squared']:.3f}")
    print(f"Median R-squared: {fq['median_r_squared']:.3f}")
    print(f"Mean correlation: {fq['mean_correlation']:.3f}")
    print(f"Median correlation: {fq['median_correlation']:.3f}")
    print(f"Mean reduced chi²: {fq['mean_reduced_chi2']:.2f}")
    
    print(f"\nGEODESIC PARAMETERS (best-fit values across all galaxies):")
    for param, stats in summary_stats['parameter_statistics'].items():
        print(f"{param:15s}: {stats['median']:6.3f} ± {stats['std']:6.3f} "
              f"(range: {stats['min']:6.3f} to {stats['max']:6.3f})")
    
    # Performance categories
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if successful_results:
        r_squared_values = [r['r_squared'] for r in successful_results.values()]
        correlations = [r['correlation'] for r in successful_results.values()]
        
        excellent_fits = [k for k, v in successful_results.items() if v['r_squared'] > 0.8]
        good_fits = [k for k, v in successful_results.items() if 0.6 < v['r_squared'] <= 0.8]
        fair_fits = [k for k, v in successful_results.items() if 0.4 < v['r_squared'] <= 0.6]
        poor_fits = [k for k, v in successful_results.items() if v['r_squared'] <= 0.4]
        
        print(f"\nFIT QUALITY BREAKDOWN:")
        print(f"Excellent fits (R² > 0.8): {len(excellent_fits):3d} galaxies ({len(excellent_fits)/len(successful_results)*100:4.1f}%)")
        print(f"Good fits (0.6 < R² ≤ 0.8): {len(good_fits):3d} galaxies ({len(good_fits)/len(successful_results)*100:4.1f}%)")
        print(f"Fair fits (0.4 < R² ≤ 0.6): {len(fair_fits):3d} galaxies ({len(fair_fits)/len(successful_results)*100:4.1f}%)")
        print(f"Poor fits (R² ≤ 0.4):       {len(poor_fits):3d} galaxies ({len(poor_fits)/len(successful_results)*100:4.1f}%)")
        
        # Top 10 best fits
        best_galaxies = sorted(successful_results.items(), 
                              key=lambda x: x[1]['r_squared'], reverse=True)[:10]
        
        print(f"\nTOP 10 BEST FITS:")
        print("Galaxy           R²      Corr    ell    alpha   Comments")
        print("-" * 60)
        for galaxy, result in best_galaxies:
            r2 = result['r_squared']
            corr = result['correlation']
            ell = result['params']['ell']
            alpha = result['params']['alpha']
            comment = "Excellent" if r2 > 0.9 else "Very good" if r2 > 0.8 else "Good"
            print(f"{galaxy:15s} {r2:5.3f}   {corr:5.3f}   {ell:5.2f}  {alpha:5.3f}   {comment}")
        
        # Worst 5 fits for comparison
        worst_galaxies = sorted(successful_results.items(), 
                               key=lambda x: x[1]['r_squared'])[:5]
        
        print(f"\nWORST 5 FITS (for comparison):")
        print("Galaxy           R²      Corr    ell    alpha   Comments")
        print("-" * 60)
        for galaxy, result in worst_galaxies:
            r2 = result['r_squared']
            corr = result['correlation']
            ell = result['params']['ell']
            alpha = result['params']['alpha']
            comment = "Poor" if r2 < 0.3 else "Fair"
            print(f"{galaxy:15s} {r2:5.3f}   {corr:5.3f}   {ell:5.2f}  {alpha:5.3f}   {comment}")
    
    print(f"\nKEY INSIGHTS:")
    
    if summary_stats['fit_quality']['mean_r_squared'] > 0.7:
        print("🎯 EXCELLENT: Geodesic theory shows strong agreement with SPARC data")
    elif summary_stats['fit_quality']['mean_r_squared'] > 0.5:
        print("📈 GOOD: Geodesic theory shows promising fit to galaxy data")
    elif summary_stats['fit_quality']['mean_r_squared'] > 0.3:
        print("🤔 FAIR: Some evidence for geodesic effects in galaxies")
    else:
        print("❌ POOR: Limited support for geodesic theory in current form")
    
    ell_stats = summary_stats['parameter_statistics']['ell']
    alpha_stats = summary_stats['parameter_statistics']['alpha']
    
    print(f"\nPhysical interpretation:")
    print(f"- Geodesic length scale clusters around {ell_stats['median']:.2f} kpc")
    print(f"- This {'matches' if 0.5 < ell_stats['median'] < 5.0 else 'differs from'} typical galaxy disk scales")
    print(f"- Geodesic coupling strength α ≈ {alpha_stats['median']:.3f}")
    print(f"- {'Strong' if alpha_stats['median'] > 0.5 else 'Moderate' if alpha_stats['median'] > 0.2 else 'Weak'} geometric enhancement needed")
    
    print(f"\n" + "=" * 70)
    print("CONCLUSION: This analysis tests whether geodesic spacetime effects")
    print("can explain galaxy rotation curves without dark matter.")
    print("=" * 70)