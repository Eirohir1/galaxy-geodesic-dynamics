# Force Diagnostic Tool - Debug Geodesic Implementation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled!")
except ImportError:
    print("GPU not available, using CPU...")
    GPU_AVAILABLE = False
    cp = np

class ForceDiagnostic:
    def __init__(self, num_stars=50, galaxy_radius=15000, cbh_mass=4.15e6):
        """
        Diagnostic tool to debug geodesic force implementation
        Using smaller number of stars for detailed analysis
        """
        self.G = 4.30091e-6  # km/s, kpc, solar mass units
        self.num_stars = num_stars
        self.galaxy_radius = galaxy_radius / 1000  # Convert to kpc
        self.cbh_mass = cbh_mass
        
        # Physical parameters
        self.base_influence_radius = 0.65 / 1000.0  # Convert to kpc
        self.influence_exponent = 0.4
        self.overlap_enhancement = 2.0
        
        self._setup_test_configuration()
        
    def _setup_test_configuration(self):
        """Set up controlled test configuration"""
        print(f"Setting up diagnostic with {self.num_stars} stars...")
        
        # Create simple radial distribution for testing
        np.random.seed(42)
        radii = np.linspace(1.0, self.galaxy_radius, self.num_stars)
        angles = np.linspace(0, 2*np.pi, self.num_stars)
        
        self.x = radii * np.cos(angles)
        self.y = radii * np.sin(angles)
        self.r = radii
        
        # Fixed stellar masses for consistent testing
        self.star_masses = np.ones(self.num_stars) * 1.5  # 1.5 solar masses each
        
        # Calculate influence radii
        self.influence_radii = self.base_influence_radius * (self.star_masses / 1.0) ** self.influence_exponent
        
        # Convert to GPU if available
        if GPU_AVAILABLE:
            self.x = cp.array(self.x)
            self.y = cp.array(self.y)
            self.star_masses = cp.array(self.star_masses)
            self.influence_radii = cp.array(self.influence_radii)
        
        print(f"Test configuration:")
        print(f"  Stars: {self.num_stars} (1.5 M☉ each)")
        print(f"  Radii: {np.min(radii):.1f} to {np.max(radii):.1f} kpc")
        print(f"  Influence: {self.influence_radii[0]*1000:.2f} pc each")
        
    def diagnose_forces(self, coupling_values=[1, 100, 1000, 10000]):
        """Detailed force diagnosis for different coupling values"""
        print("\n🔬 FORCE DIAGNOSTIC ANALYSIS")
        print("="*50)
        
        results = []
        
        for coupling in coupling_values:
            print(f"\nAnalyzing coupling = {coupling}")
            
            # Calculate forces
            if GPU_AVAILABLE:
                x, y = cp.array(self.x), cp.array(self.y)
            else:
                x, y = self.x.copy(), self.y.copy()
            
            # Central forces
            r_central = cp.sqrt(x**2 + y**2 + 0.001**2) if GPU_AVAILABLE else np.sqrt(x**2 + y**2 + 0.001**2)
            a_central_mag = self.G * self.cbh_mass / r_central**2
            
            # Geodesic forces with detailed diagnostics
            ax_geo, ay_geo, diagnostics = self._calculate_geodesic_forces_detailed(x, y, coupling)
            
            if GPU_AVAILABLE:
                a_central_mag = cp.asnumpy(a_central_mag)
                a_geo_mag = cp.asnumpy(cp.sqrt(ax_geo**2 + ay_geo**2))
                r_central = cp.asnumpy(r_central)
            else:
                a_geo_mag = np.sqrt(ax_geo**2 + ay_geo**2)
            
            # Calculate force ratios
            force_ratios = a_geo_mag / a_central_mag
            
            # Summary statistics
            max_central = np.max(a_central_mag)
            max_geodesic = np.max(a_geo_mag)
            max_ratio = np.max(force_ratios)
            mean_ratio = np.mean(force_ratios)
            
            print(f"  Max central force: {max_central:.2e}")
            print(f"  Max geodesic force: {max_geodesic:.2e}")
            print(f"  Max force ratio: {max_ratio:.2e}")
            print(f"  Mean force ratio: {mean_ratio:.2e}")
            print(f"  Overlaps detected: {diagnostics['overlaps_detected']}")
            print(f"  Total interactions: {diagnostics['total_interactions']}")
            
            results.append({
                'coupling': coupling,
                'radii': r_central.copy(),
                'central_forces': a_central_mag.copy(),
                'geodesic_forces': a_geo_mag.copy(),
                'force_ratios': force_ratios.copy(),
                'max_ratio': max_ratio,
                'mean_ratio': mean_ratio,
                'diagnostics': diagnostics
            })
        
        return results
    
    def _calculate_geodesic_forces_detailed(self, x, y, coupling):
        """Calculate geodesic forces with detailed diagnostics"""
        if GPU_AVAILABLE:
            return self._geodesic_forces_gpu_detailed(x, y, coupling)
        else:
            return self._geodesic_forces_cpu_detailed(x, y, coupling)
    
    def _geodesic_forces_gpu_detailed(self, x, y, coupling):
        """GPU geodesic forces with diagnostics"""
        n = len(x)
        
        # Distance matrix
        x_diff = x[:, cp.newaxis] - x[cp.newaxis, :]
        y_diff = y[:, cp.newaxis] - y[cp.newaxis, :]
        r_matrix = cp.sqrt(x_diff**2 + y_diff**2 + 1e-6)
        
        # Influence matrix
        influence_matrix = self.influence_radii[cp.newaxis, :] * cp.ones((n, n))
        within_influence = r_matrix <= influence_matrix
        
        # Count overlaps
        overlaps_detected = int(cp.sum(within_influence) - n)  # Subtract diagonal
        
        # Well depth calculation with detailed logging
        well_depth = cp.where(
            within_influence,
            self.star_masses[cp.newaxis, :] * cp.exp(-r_matrix / influence_matrix),
            0.0
        )
        
        # Remove self-interaction
        cp.fill_diagonal(well_depth, 0.0)
        cp.fill_diagonal(r_matrix, cp.inf)
        
        # Check if well_depth has any significant values
        max_well_depth = float(cp.max(well_depth))
        
        # Overlap enhancement
        overlap_count = cp.sum(within_influence, axis=1)
        overlap_factor = 1.0 + (overlap_count - 1) * (self.overlap_enhancement - 1.0) / 10.0
        
        # Force calculation
        force_magnitude = coupling * self.G * well_depth / (r_matrix**2)
        force_magnitude *= overlap_factor[:, cp.newaxis]
        
        # Check force magnitudes
        max_force_magnitude = float(cp.max(force_magnitude))
        
        # Force components
        fx = force_magnitude * x_diff / r_matrix
        fy = force_magnitude * y_diff / r_matrix
        
        # Replace any NaN values
        fx = cp.nan_to_num(fx)
        fy = cp.nan_to_num(fy)
        
        # Sum forces (attractive)
        ax = -cp.sum(fx, axis=1)
        ay = -cp.sum(fy, axis=1)
        
        diagnostics = {
            'overlaps_detected': overlaps_detected,
            'total_interactions': n * (n - 1),
            'max_well_depth': max_well_depth,
            'max_force_magnitude': max_force_magnitude,
            'overlap_factor_range': (float(cp.min(overlap_factor)), float(cp.max(overlap_factor)))
        }
        
        return ax, ay, diagnostics
    
    def _geodesic_forces_cpu_detailed(self, x, y, coupling):
        """CPU geodesic forces with diagnostics"""
        n = len(x)
        ax = np.zeros(n)
        ay = np.zeros(n)
        
        overlaps_detected = 0
        max_well_depth = 0.0
        max_force_magnitude = 0.0
        
        for i in range(n):
            overlap_count = 0
            
            for j in range(n):
                if i != j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = np.sqrt(dx**2 + dy**2 + 1e-6)
                    
                    if dr <= self.influence_radii[j]:
                        overlaps_detected += 1
                        overlap_count += 1
                        
                        well_depth = self.star_masses[j] * np.exp(-dr / self.influence_radii[j])
                        max_well_depth = max(max_well_depth, well_depth)
                        
                        overlap_factor = 1.0 + overlap_count * (self.overlap_enhancement - 1.0) / 10.0
                        force_mag = coupling * self.G * well_depth * overlap_factor / dr**2
                        max_force_magnitude = max(max_force_magnitude, force_mag)
                        
                        ax[i] -= force_mag * dx / dr
                        ay[i] -= force_mag * dy / dr
        
        diagnostics = {
            'overlaps_detected': overlaps_detected,
            'total_interactions': n * (n - 1),
            'max_well_depth': max_well_depth,
            'max_force_magnitude': max_force_magnitude,
            'overlap_factor_range': (1.0, 2.0)  # Simplified for CPU
        }
        
        return ax, ay, diagnostics
    
    def plot_force_analysis(self, results):
        """Create comprehensive force analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Force magnitudes vs radius
        ax1 = axes[0, 0]
        for result in results:
            coupling = result['coupling']
            radii = result['radii']
            central = result['central_forces']
            geodesic = result['geodesic_forces']
            
            ax1.loglog(radii, central, 'k--', alpha=0.7, label='Central (all)')
            ax1.loglog(radii, geodesic, label=f'Geodesic (c={coupling})')
        
        ax1.set_xlabel('Radius (kpc)')
        ax1.set_ylabel('Force Magnitude')
        ax1.set_title('Force Magnitudes vs Radius')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Force ratios
        ax2 = axes[0, 1]
        for result in results:
            coupling = result['coupling']
            radii = result['radii']
            ratios = result['force_ratios']
            
            ax2.semilogx(radii, ratios, 'o-', label=f'c={coupling}')
        
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal strength')
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_ylabel('Geodesic/Central Force Ratio')
        ax2.set_title('Force Ratio Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Coupling effectiveness
        ax3 = axes[0, 2]
        couplings = [r['coupling'] for r in results]
        max_ratios = [r['max_ratio'] for r in results]
        mean_ratios = [r['mean_ratio'] for r in results]
        
        ax3.loglog(couplings, max_ratios, 'ro-', label='Max ratio')
        ax3.loglog(couplings, mean_ratios, 'bo-', label='Mean ratio')
        ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Equal strength')
        ax3.set_xlabel('Coupling Strength')
        ax3.set_ylabel('Force Ratio')
        ax3.set_title('Coupling Effectiveness')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Diagnostic summary
        ax4 = axes[1, 0]
        overlaps = [r['diagnostics']['overlaps_detected'] for r in results]
        max_wells = [r['diagnostics']['max_well_depth'] for r in results]
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(couplings, overlaps, 'go-', label='Overlaps detected')
        line2 = ax4_twin.semilogy(couplings, max_wells, 'mo-', label='Max well depth')
        
        ax4.set_xlabel('Coupling')
        ax4.set_ylabel('Number of Overlaps', color='g')
        ax4_twin.set_ylabel('Max Well Depth', color='m')
        ax4.set_title('Geodesic Interaction Diagnostics')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='best')
        
        # 5. Star configuration
        ax5 = axes[1, 1]
        if GPU_AVAILABLE:
            x_plot = cp.asnumpy(self.x)
            y_plot = cp.asnumpy(self.y)
            influence_plot = cp.asnumpy(self.influence_radii)
        else:
            x_plot = self.x
            y_plot = self.y
            influence_plot = self.influence_radii
        
        # Plot stars and their influence spheres
        ax5.scatter(x_plot, y_plot, s=50, c='blue', alpha=0.7)
        
        # Draw influence circles for first few stars
        for i in range(min(10, len(x_plot))):
            circle = patches.Circle((x_plot[i], y_plot[i]), influence_plot[i], 
                                  fill=False, edgecolor='red', alpha=0.3)
            ax5.add_patch(circle)
        
        # Galaxy boundary
        galaxy_circle = patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                                     edgecolor='black', linestyle='--', alpha=0.5)
        ax5.add_patch(galaxy_circle)
        
        ax5.set_xlim(-self.galaxy_radius*1.1, self.galaxy_radius*1.1)
        ax5.set_ylim(-self.galaxy_radius*1.1, self.galaxy_radius*1.1)
        ax5.set_xlabel('x (kpc)')
        ax5.set_ylabel('y (kpc)')
        ax5.set_title('Test Configuration\n(Red circles = influence spheres)')
        ax5.set_aspect('equal')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Find most effective coupling
        best_idx = np.argmax(max_ratios)
        best_coupling = couplings[best_idx]
        best_ratio = max_ratios[best_idx]
        
        summary_text = f"""
FORCE DIAGNOSTIC SUMMARY

CONFIGURATION:
• {self.num_stars} test stars (1.5 M☉ each)
• Galaxy radius: {self.galaxy_radius:.1f} kpc
• Central BH: {self.cbh_mass:.2e} M☉
• Influence radius: {influence_plot[0]*1000:.2f} pc

COUPLING ANALYSIS:
• Range tested: {min(couplings)} - {max(couplings):,}
• Best coupling: {best_coupling:,}
• Best force ratio: {best_ratio:.2e}

INTERACTION STATISTICS:
• Overlaps detected: {overlaps[best_idx]} 
• Total possible: {results[best_idx]['diagnostics']['total_interactions']}
• Overlap rate: {overlaps[best_idx]/results[best_idx]['diagnostics']['total_interactions']:.1%}

DIAGNOSIS:
{'✅ Geodesic forces are working!' if best_ratio > 0.01 
 else '⚠️ Geodesic forces are very weak' if best_ratio > 1e-6
 else '❌ Geodesic forces are negligible'}

PROBLEM IDENTIFICATION:
{'Forces too weak - need stronger coupling' if best_ratio < 0.01
 else 'Implementation working correctly'}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('force_diagnostic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_coupling, best_ratio

def run_force_diagnostic():
    """Run comprehensive force diagnostic"""
    print("🔬 FORCE DIAGNOSTIC TOOL")
    print("Debugging geodesic force implementation")
    
    # Create diagnostic tool
    diagnostic = ForceDiagnostic(num_stars=50, galaxy_radius=15000, cbh_mass=4.15e6)
    
    # Run force analysis
    results = diagnostic.diagnose_forces([1, 100, 1000, 10000, 50000])
    
    # Plot analysis
    best_coupling, best_ratio = diagnostic.plot_force_analysis(results)
    
    print(f"\n🎯 DIAGNOSTIC COMPLETE!")
    print(f"Best coupling: {best_coupling:,}")
    print(f"Best force ratio: {best_ratio:.2e}")
    
    if best_ratio > 0.01:
        print("\n✅ Geodesic forces are significant and working correctly!")
        print("The parameter sweep should show improvement with this coupling.")
    elif best_ratio > 1e-6:
        print("\n⚠️ Geodesic forces are very weak but detectable.")
        print("Need much stronger coupling or different implementation.")
    else:
        print("\n❌ Geodesic forces are negligible.")
        print("Fundamental implementation problem identified.")
    
    return results

if __name__ == "__main__":
    # Run the diagnostic
    diagnostic_results = run_force_diagnostic()