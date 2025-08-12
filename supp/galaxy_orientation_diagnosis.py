#!/usr/bin/env python3
"""
Diagnose why large galaxies are failing
Check: orientation, velocity profiles, scaling assumptions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
import glob
import os
from pathlib import Path

def read_sparc_file(filepath):
    """Read a single SPARC rotmod file with full column info."""
    try:
        data = pd.read_csv(filepath, sep=r'\s+', comment='#',
                          names=['Rad', 'Vobs', 'errV', 'Vgas', 'Vdisk', 'Vbul', 'SBdisk', 'SBbul'])
        
        data = data.dropna(subset=['Rad', 'Vobs'])
        data = data[(data['Rad'] > 0) & (data['Vobs'] > 0)]
        
        return data if len(data) >= 3 else None
        
    except Exception:
        return None

def analyze_galaxy_properties(data, galaxy_name):
    """
    Analyze galaxy properties to understand why fits might fail.
    """
    
    radius = data['Rad'].values
    velocity = data['Vobs'].values
    
    # Basic properties
    props = {
        'name': galaxy_name,
        'n_points': len(radius),
        'r_min': np.min(radius),
        'r_max': np.max(radius),
        'r_range': np.max(radius) - np.min(radius),
        'v_min': np.min(velocity),
        'v_max': np.max(velocity),
        'v_range': np.max(velocity) - np.min(velocity),
        'v_mean': np.mean(velocity),
    }
    
    # Rotation curve shape analysis
    if len(velocity) > 3:
        # Is the curve rising, flat, or declining?
        v_inner = np.mean(velocity[:len(velocity)//3])
        v_outer = np.mean(velocity[-len(velocity)//3:])
        
        props['v_inner'] = v_inner
        props['v_outer'] = v_outer
        props['curve_type'] = 'rising' if v_outer > v_inner * 1.1 else 'flat' if v_outer > v_inner * 0.9 else 'declining'
        
        # Velocity gradient
        dv_dr = np.gradient(velocity, radius)
        props['max_gradient'] = np.max(np.abs(dv_dr))
        props['mean_gradient'] = np.mean(np.abs(dv_dr))
    
    # Check for baryonic components
    if 'Vgas' in data.columns:
        props['has_gas'] = not data['Vgas'].isna().all()
        props['has_disk'] = not data['Vdisk'].isna().all()
        props['has_bulge'] = not data['Vbul'].isna().all()
        
        if props['has_gas']:
            props['v_gas_max'] = np.nanmax(data['Vgas'])
        if props['has_disk']:
            props['v_disk_max'] = np.nanmax(data['Vdisk'])
        if props['has_bulge']:
            props['v_bulge_max'] = np.nanmax(data['Vbul'])
    
    # Galaxy type indicators
    name_upper = galaxy_name.upper()
    props['is_ngc'] = name_upper.startswith('NGC')
    props['is_ugc'] = name_upper.startswith('UGC')
    props['is_ddo'] = name_upper.startswith('DDO')
    props['is_edge_on_candidate'] = name_upper.startswith('F') or 'ESO' in name_upper
    
    # Velocity scale classification
    if props['v_max'] < 50:
        props['velocity_class'] = 'low'
    elif props['v_max'] < 150:
        props['velocity_class'] = 'medium'
    else:
        props['velocity_class'] = 'high'
    
    return props

def plot_problem_galaxies(galaxy_names, sparc_directory, output_dir='problem_diagnosis'):
    """
    Plot the problematic galaxies to see what's happening.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    galaxy_props = []
    
    for i, galaxy_name in enumerate(galaxy_names):
        if i >= 9:  # Only plot first 9
            break
            
        filepath = os.path.join(sparc_directory, f"{galaxy_name}_rotmod.dat")
        
        if not os.path.exists(filepath):
            continue
            
        data = read_sparc_file(filepath)
        if data is None:
            continue
        
        # Analyze properties
        props = analyze_galaxy_properties(data, galaxy_name)
        galaxy_props.append(props)
        
        # Plot rotation curve
        ax = axes[i]
        
        radius = data['Rad'].values
        velocity = data['Vobs'].values
        
        # Plot observed data
        if 'errV' in data.columns and not data['errV'].isna().all():
            ax.errorbar(radius, velocity, yerr=data['errV'], fmt='ko', alpha=0.7, label='Observed')
        else:
            ax.plot(radius, velocity, 'ko', alpha=0.7, label='Observed')
        
        # Plot baryonic components if available
        if 'Vgas' in data.columns and not data['Vgas'].isna().all():
            ax.plot(radius, data['Vgas'], 'b--', alpha=0.7, label='Gas')
        if 'Vdisk' in data.columns and not data['Vdisk'].isna().all():
            ax.plot(radius, data['Vdisk'], 'g--', alpha=0.7, label='Disk')
        if 'Vbul' in data.columns and not data['Vbul'].isna().all():
            ax.plot(radius, data['Vbul'], 'r--', alpha=0.7, label='Bulge')
        
        # Title with key info
        title = f"{galaxy_name}\n"
        title += f"V: {props['v_min']:.0f}-{props['v_max']:.0f} km/s, "
        title += f"R: {props['r_max']:.1f} kpc\\n"
        title += f"Shape: {props.get('curve_type', 'unknown')}"
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(len(galaxy_names), 9):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'problem_galaxies_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return galaxy_props

def compare_galaxy_classes():
    """
    Compare properties of successful vs failed galaxies.
    """
    
    sparc_directory = r"C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts"
    
    # Define galaxy sets based on previous results
    successful_galaxies = ['CamB', 'D564-8', 'UGC06628', 'UGC01281', 'DDO154']
    failed_galaxies = ['NGC0891', 'NGC2955', 'NGC3893', 'NGC5005']
    
    print("GALAXY ORIENTATION AND SCALING DIAGNOSIS")
    print("=" * 70)
    
    # Analyze successful galaxies
    print("\\nSUCCESSFUL GALAXIES:")
    print("-" * 30)
    
    successful_props = []
    for galaxy_name in successful_galaxies:
        filepath = os.path.join(sparc_directory, f"{galaxy_name}_rotmod.dat")
        if os.path.exists(filepath):
            data = read_sparc_file(filepath)
            if data is not None:
                props = analyze_galaxy_properties(data, galaxy_name)
                successful_props.append(props)
                
                print(f"{galaxy_name:12s}: V = {props['v_min']:3.0f}-{props['v_max']:3.0f} km/s, "
                      f"R = {props['r_max']:4.1f} kpc, "
                      f"Shape = {props.get('curve_type', 'unknown'):8s}, "
                      f"Class = {props['velocity_class']:6s}")
    
    # Analyze failed galaxies
    print("\\nFAILED GALAXIES:")
    print("-" * 30)
    
    failed_props = []
    for galaxy_name in failed_galaxies:
        filepath = os.path.join(sparc_directory, f"{galaxy_name}_rotmod.dat")
        if os.path.exists(filepath):
            data = read_sparc_file(filepath)
            if data is not None:
                props = analyze_galaxy_properties(data, galaxy_name)
                failed_props.append(props)
                
                print(f"{galaxy_name:12s}: V = {props['v_min']:3.0f}-{props['v_max']:3.0f} km/s, "
                      f"R = {props['r_max']:4.1f} kpc, "
                      f"Shape = {props.get('curve_type', 'unknown'):8s}, "
                      f"Class = {props['velocity_class']:6s}")
    
    # Statistical comparison
    print("\\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)
    
    if successful_props and failed_props:
        # Velocity comparison
        succ_v_max = [p['v_max'] for p in successful_props]
        fail_v_max = [p['v_max'] for p in failed_props]
        
        print(f"Maximum velocity:")
        print(f"  Successful: {np.mean(succ_v_max):5.1f} ± {np.std(succ_v_max):4.1f} km/s")
        print(f"  Failed:     {np.mean(fail_v_max):5.1f} ± {np.std(fail_v_max):4.1f} km/s")
        
        # Size comparison
        succ_r_max = [p['r_max'] for p in successful_props]
        fail_r_max = [p['r_max'] for p in failed_props]
        
        print(f"Maximum radius:")
        print(f"  Successful: {np.mean(succ_r_max):5.1f} ± {np.std(succ_r_max):4.1f} kpc")
        print(f"  Failed:     {np.mean(fail_r_max):5.1f} ± {np.std(fail_r_max):4.1f} kpc")
        
        # Galaxy type analysis
        succ_ngc = sum(1 for p in successful_props if p['is_ngc'])
        fail_ngc = sum(1 for p in failed_props if p['is_ngc'])
        
        print(f"Galaxy types:")
        print(f"  Successful NGC: {succ_ngc}/{len(successful_props)} ({succ_ngc/len(successful_props)*100:.0f}%)")
        print(f"  Failed NGC:     {fail_ngc}/{len(failed_props)} ({fail_ngc/len(failed_props)*100:.0f}%)")
        
        # Curve shape analysis
        succ_shapes = [p.get('curve_type', 'unknown') for p in successful_props]
        fail_shapes = [p.get('curve_type', 'unknown') for p in failed_props]
        
        print(f"Rotation curve shapes:")
        for shape in ['rising', 'flat', 'declining']:
            succ_count = succ_shapes.count(shape)
            fail_count = fail_shapes.count(shape)
            print(f"  {shape:9s}: Successful {succ_count}/{len(successful_props)}, Failed {fail_count}/{len(failed_props)}")
    
    # Plot the problematic cases
    print(f"\\nGenerating diagnostic plots...")
    
    print("\\nPlotting successful galaxies...")
    successful_props_plot = plot_problem_galaxies(successful_galaxies, sparc_directory, 'successful_galaxies')
    
    print("Plotting failed galaxies...")
    failed_props_plot = plot_problem_galaxies(failed_galaxies, sparc_directory, 'failed_galaxies')
    
    print(f"\\nPLOTS SAVED:")
    print(f"  successful_galaxies/problem_galaxies_overview.png")
    print(f"  failed_galaxies/problem_galaxies_overview.png")
    
    # Key insights
    print(f"\\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    
    if successful_props and failed_props:
        avg_succ_v = np.mean([p['v_max'] for p in successful_props])
        avg_fail_v = np.mean([p['v_max'] for p in failed_props])
        
        if avg_fail_v > avg_succ_v * 2:
            print("🔍 VELOCITY THRESHOLD: Failed galaxies have much higher velocities")
            print(f"   Cutoff appears to be around {avg_succ_v * 1.5:.0f} km/s")
        
        fail_ngc_fraction = fail_ngc / len(failed_props) if failed_props else 0
        if fail_ngc_fraction > 0.7:
            print("🔍 GALAXY TYPE: Most failures are NGC galaxies (major spirals)")
            print("   These might be edge-on or have different physics")
        
        # Check if failed galaxies have different baryonic structure
        print("\\n🔍 NEXT STEPS TO INVESTIGATE:")
        print("1. Check if NGC failures are edge-on galaxies")
        print("2. Examine if high-velocity systems need different geodesic physics")
        print("3. Test if the model works better with velocity-dependent parameters")
        print("4. Consider if massive galaxies need dark matter + geodesic hybrid")

if __name__ == "__main__":
    try:
        compare_galaxy_classes()
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()