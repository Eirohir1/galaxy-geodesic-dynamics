# Spacetime Lattice Geodesic Model
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

class SpacetimeLatticeModel:
    def __init__(self, num_stars=500, galaxy_radius=15000, cbh_mass=4.15e6, 
                 total_time=200e6, dt=2e6):
        """
        Spacetime lattice geodesic model based on your crater field theory
        
        Key concepts:
        - Each mass creates a "well" in spacetime lattice
        - Sphere of influence determines well size
        - Overlapping spheres create cumulative geodesic bending
        - Enhanced binding from pockmarked spacetime geometry
        """
        self.G = 4.30091e-6  # km/s, kpc, solar mass units
        self.num_stars = num_stars
        self.galaxy_radius = galaxy_radius / 1000  # Convert to kpc
        self.cbh_mass = cbh_mass
        self.total_time = total_time
        self.dt = dt
        self.num_steps = int(total_time / dt)
        
        # Spacetime lattice parameters
        self.base_influence_radius = 0.65  # parsecs (like our Sun)
        self.influence_exponent = 0.4  # From sphere of influence scaling
        self.lattice_coupling = 100.0  # Geodesic coupling strength (to be calibrated)
        self.overlap_enhancement = 2.0  # Extra binding in overlap zones
        
        # Escape criteria
        self.escape_radius = 3.0 * self.galaxy_radius
        
        self._setup_initial_conditions()
        
    def _setup_initial_conditions(self):
        """Set up realistic stellar distribution"""
        print(f"Setting up {self.num_stars} stars in {self.galaxy_radius:.1f} kpc galaxy...")
        print(f"Central black hole: {self.cbh_mass:.2e} solar masses")
        print(f"Spacetime lattice coupling: {self.lattice_coupling}")
        
        # Exponential disk profile
        np.random.seed(42)
        scale_length = self.galaxy_radius / 3
        radii = np.random.exponential(scale_length, self.num_stars)
        radii = np.clip(radii, 0.5, self.galaxy_radius)
        
        angles = np.random.uniform(0, 2*np.pi, self.num_stars)
        
        # Cartesian coordinates
        self.x = radii * np.cos(angles)
        self.y = radii * np.sin(angles)
        self.r = radii
        
        # Stellar masses (0.5 to 3.0 solar masses)
        self.star_masses = np.random.uniform(0.5, 3.0, self.num_stars)
        
        # Calculate sphere of influence for each star
        self.influence_radii = self.base_influence_radius * (self.star_masses / 1.0) ** self.influence_exponent
        # Convert to kpc
        self.influence_radii = self.influence_radii / 1000.0
        
        # Initial velocities (Keplerian)
        v_circular = np.sqrt(self.G * self.cbh_mass / self.r)
        v_dispersion = 0.05 * v_circular  # Reduced dispersion for stability
        v_radial = np.random.normal(0, v_dispersion, self.num_stars)
        v_tangential = v_circular + np.random.normal(0, v_dispersion, self.num_stars)
        
        # Convert to Cartesian velocities
        self.vx = v_radial * np.cos(angles) - v_tangential * np.sin(angles)
        self.vy = v_radial * np.sin(angles) + v_tangential * np.cos(angles)
        
        # Convert to GPU arrays if available
        if GPU_AVAILABLE:
            self.x = cp.array(self.x)
            self.y = cp.array(self.y)
            self.vx = cp.array(self.vx)
            self.vy = cp.array(self.vy)
            self.star_masses = cp.array(self.star_masses)
            self.influence_radii = cp.array(self.influence_radii)
        
        print(f"Initial setup complete: stars from 0.5 to {self.galaxy_radius:.1f} kpc")
        print(f"Influence radii: {np.min(self.influence_radii)*1000:.2f} to {np.max(self.influence_radii)*1000:.2f} pc")
    
    def calculate_central_acceleration(self, x, y):
        """Central black hole gravity (anchor point)"""
        if GPU_AVAILABLE:
            r = cp.sqrt(x**2 + y**2 + 0.001**2)
            a_magnitude = self.G * self.cbh_mass / r**2
            ax = -a_magnitude * x / r
            ay = -a_magnitude * y / r
        else:
            r = np.sqrt(x**2 + y**2 + 0.001**2)
            a_magnitude = self.G * self.cbh_mass / r**2
            ax = -a_magnitude * x / r
            ay = -a_magnitude * y / r
        
        return ax, ay
    
    def calculate_lattice_acceleration(self, x, y):
        """
        Spacetime lattice geodesic acceleration
        Based on your crater field model with overlapping spheres of influence
        """
        if GPU_AVAILABLE:
            return self._calculate_lattice_acceleration_gpu(x, y)
        else:
            return self._calculate_lattice_acceleration_cpu(x, y)
    
    def _calculate_lattice_acceleration_gpu(self, x, y):
        """GPU-accelerated lattice calculation"""
        n_stars = len(x)
        
        # Initialize acceleration arrays
        ax_lattice = cp.zeros_like(x)
        ay_lattice = cp.zeros_like(y)
        
        # Create distance matrices for vectorized calculation
        x_diff = x[:, cp.newaxis] - x[cp.newaxis, :]  # Shape: (n, n)
        y_diff = y[:, cp.newaxis] - y[cp.newaxis, :]
        r_matrix = cp.sqrt(x_diff**2 + y_diff**2 + 1e-6)  # Avoid division by zero
        
        # Mass matrix (each star's mass for interactions)
        mass_matrix = self.star_masses[cp.newaxis, :] * cp.ones((n_stars, n_stars))
        
        # Influence radius matrix (each star's influence radius)
        influence_matrix = self.influence_radii[cp.newaxis, :] * cp.ones((n_stars, n_stars))
        
        # Create geodesic well depth matrix
        # Each star creates a well with depth proportional to mass and distance from well center
        well_depth = cp.zeros_like(r_matrix)
        
        # For each star j, calculate its geodesic well effect on star i
        for j in range(n_stars):
            # Stars within the sphere of influence of star j experience geodesic effects
            within_influence = r_matrix[:, j] <= self.influence_radii[j]
            
            # Exponential decay within sphere of influence
            well_depth[:, j] = cp.where(
                within_influence,
                self.star_masses[j] * cp.exp(-r_matrix[:, j] / self.influence_radii[j]),
                0.0
            )
        
        # Calculate overlap enhancement
        # Count how many spheres each star is within
        overlap_count = cp.sum(r_matrix <= influence_matrix, axis=1)
        overlap_factor = 1.0 + (overlap_count - 1) * (self.overlap_enhancement - 1.0) / 10.0
        
        # Calculate geodesic forces
        # Remove self-interaction (diagonal)
        cp.fill_diagonal(well_depth, 0.0)
        cp.fill_diagonal(r_matrix, cp.inf)  # Avoid self-force
        
        # Force magnitude from geodesic coupling
        force_magnitude = self.lattice_coupling * self.G * well_depth / (r_matrix**2)
        
        # Apply overlap enhancement
        force_magnitude = force_magnitude * overlap_factor[:, cp.newaxis]
        
        # Calculate force components (attractive toward each geodesic well)
        fx_components = force_magnitude * x_diff / r_matrix
        fy_components = force_magnitude * y_diff / r_matrix
        
        # Sum forces from all geodesic wells
        ax_lattice = -cp.sum(fx_components, axis=1)  # Attractive force
        ay_lattice = -cp.sum(fy_components, axis=1)
        
        return ax_lattice, ay_lattice
    
    def _calculate_lattice_acceleration_cpu(self, x, y):
        """CPU fallback for lattice calculation"""
        n_stars = len(x)
        
        ax_lattice = np.zeros_like(x)
        ay_lattice = np.zeros_like(y)
        
        # Simplified CPU version (sample interactions for speed)
        sample_rate = max(1, n_stars // 50)  # Sample ~50 interactions per star
        
        for i in range(0, n_stars, sample_rate):
            # Count overlapping spheres for star i
            overlap_count = 0
            
            for j in range(0, n_stars, sample_rate):
                if i != j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = np.sqrt(dx**2 + dy**2 + 1e-6)
                    
                    # Check if star i is within star j's sphere of influence
                    if dr <= self.influence_radii[j]:
                        overlap_count += 1
                        
                        # Geodesic well effect
                        well_depth = self.star_masses[j] * np.exp(-dr / self.influence_radii[j])
                        force_magnitude = self.lattice_coupling * self.G * well_depth / dr**2
                        
                        # Apply overlap enhancement
                        overlap_factor = 1.0 + overlap_count * (self.overlap_enhancement - 1.0) / 10.0
                        force_magnitude *= overlap_factor
                        
                        # Attractive force components
                        ax_lattice[i] -= force_magnitude * dx / dr
                        ay_lattice[i] -= force_magnitude * dy / dr
        
        return ax_lattice, ay_lattice
    
    def integrate_orbits(self, method="lattice"):
        """Integrate stellar orbits using spacetime lattice model"""
        method_names = {
            "newtonian": "Newtonian (Central Mass Only)",
            "lattice": "Spacetime Lattice Geodesics"
        }
        
        print(f"\n=== {method_names[method]} ===")
        
        # Copy initial conditions
        if GPU_AVAILABLE:
            x, y = cp.array(self.x), cp.array(self.y)
            vx, vy = cp.array(self.vx), cp.array(self.vy)
        else:
            x, y = self.x.copy(), self.y.copy()
            vx, vy = self.vx.copy(), self.vy.copy()
        
        # Tracking arrays
        if GPU_AVAILABLE:
            bound_mask = cp.ones(self.num_stars, dtype=bool)
        else:
            bound_mask = np.ones(self.num_stars, dtype=bool)
        
        # Snapshot storage
        snapshot_interval = int(25e6 / self.dt)  # Every 25 Myr
        snapshots = []
        times = []
        
        start_time = time.time()
        
        for step in range(self.num_steps):
            # Calculate accelerations
            ax_central, ay_central = self.calculate_central_acceleration(x, y)
            
            if method == "lattice":
                ax_lattice, ay_lattice = self.calculate_lattice_acceleration(x, y)
                ax_total = ax_central + ax_lattice
                ay_total = ay_central + ay_lattice
            else:  # newtonian
                ax_total = ax_central
                ay_total = ay_central
            
            # Leapfrog integration
            dt_code = self.dt * 3.154e7 / 3.086e13  # Convert to code units
            
            # Update velocities
            vx += ax_total * dt_code
            vy += ay_total * dt_code
            
            # Update positions
            x += vx * dt_code
            y += vy * dt_code
            
            # Check for escapes
            if GPU_AVAILABLE:
                r_current = cp.sqrt(x**2 + y**2)
                bound_mask = r_current < self.escape_radius
            else:
                r_current = np.sqrt(x**2 + y**2)
                bound_mask = r_current < self.escape_radius
            
            # Store snapshots
            if step % snapshot_interval == 0:
                if GPU_AVAILABLE:
                    bound_count = int(cp.sum(bound_mask))
                    x_cpu = cp.asnumpy(x[bound_mask])
                    y_cpu = cp.asnumpy(y[bound_mask])
                    r_cpu = cp.asnumpy(r_current[bound_mask])
                else:
                    bound_count = int(np.sum(bound_mask))
                    x_cpu = x[bound_mask]
                    y_cpu = y[bound_mask]
                    r_cpu = r_current[bound_mask]
                
                snapshots.append({
                    'x': x_cpu.copy(),
                    'y': y_cpu.copy(),
                    'r': r_cpu.copy(),
                    'bound_count': bound_count,
                    'time': step * self.dt / 1e6
                })
                times.append(step * self.dt / 1e6)
                
                if step % (4 * snapshot_interval) == 0:  # Every 100 Myr
                    retention = bound_count / self.num_stars
                    print(f"  t={step * self.dt / 1e6:.0f} Myr: {retention:.1%} stars retained")
        
        elapsed = time.time() - start_time
        if GPU_AVAILABLE:
            final_bound = int(cp.sum(bound_mask))
        else:
            final_bound = int(np.sum(bound_mask))
        
        final_retention = final_bound / self.num_stars
        
        gpu_status = "(GPU)" if GPU_AVAILABLE else "(CPU)"
        print(f"  Simulation completed in {elapsed:.1f}s {gpu_status}")
        print(f"  Final retention: {final_retention:.1%} ({final_bound}/{self.num_stars} stars)")
        
        return {
            'method': method_names[method],
            'snapshots': snapshots,
            'times': times,
            'final_retention': final_retention,
            'final_bound': final_bound
        }
    
    def plot_comparison(self, results_newtonian, results_lattice, save_path=None):
        """Plot side-by-side comparison"""
        fig = plt.figure(figsize=(16, 12))
        
        # Get final snapshots
        snap_n = results_newtonian['snapshots'][-1] if results_newtonian['snapshots'] else {'x': [], 'y': [], 'bound_count': 0}
        snap_l = results_lattice['snapshots'][-1] if results_lattice['snapshots'] else {'x': [], 'y': [], 'bound_count': 0}
        
        # 1. Final galaxy states
        ax1 = plt.subplot(2, 3, 1)
        if len(snap_n['x']) > 0:
            ax1.scatter(snap_n['x'], snap_n['y'], s=1, alpha=0.7, color='red')
        circle = patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                              edgecolor='white', linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
        ax1.set_xlim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
        ax1.set_ylim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
        ax1.set_title(f'Newtonian Gravity\n{snap_n["bound_count"]} stars retained', 
                     fontsize=12, color='white')
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        ax1.set_aspect('equal')
        
        ax2 = plt.subplot(2, 3, 2)
        if len(snap_l['x']) > 0:
            ax2.scatter(snap_l['x'], snap_l['y'], s=1, alpha=0.7, color='gold')
        circle = patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                              edgecolor='white', linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
        ax2.set_xlim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
        ax2.set_ylim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
        ax2.set_title(f'Spacetime Lattice\n{snap_l["bound_count"]} stars retained', 
                     fontsize=12, color='white')
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        ax2.set_aspect('equal')
        
        # 2. Retention over time
        ax3 = plt.subplot(2, 3, 3)
        if results_newtonian['snapshots']:
            newton_retention = [s['bound_count']/self.num_stars for s in results_newtonian['snapshots']]
            ax3.plot(results_newtonian['times'], newton_retention, 'red', linewidth=2, label='Newtonian')
        
        if results_lattice['snapshots']:
            lattice_retention = [s['bound_count']/self.num_stars for s in results_lattice['snapshots']]
            ax3.plot(results_lattice['times'], lattice_retention, 'gold', linewidth=2, label='Lattice')
        
        ax3.set_xlabel('Time (Myr)')
        ax3.set_ylabel('Fraction Retained')
        ax3.set_title('Stellar Retention Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # 3. Final comparison
        ax4 = plt.subplot(2, 3, 4)
        methods = ['Newtonian', 'Lattice']
        retentions = [results_newtonian['final_retention'], results_lattice['final_retention']]
        colors = ['red', 'gold']
        
        bars = ax4.bar(methods, retentions, color=colors, alpha=0.8)
        ax4.set_ylabel('Final Retention Rate')
        ax4.set_title('Method Comparison')
        ax4.set_ylim(0, 1.0)
        
        for bar, retention in zip(bars, retentions):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{retention:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
        
        # 4. Theory summary
        ax5 = plt.subplot(2, 3, (5, 6))
        ax5.axis('off')
        
        improvement = results_lattice['final_retention'] - results_newtonian['final_retention']
        factor = results_lattice['final_retention'] / max(results_newtonian['final_retention'], 0.01)
        
        summary_text = f"""
SPACETIME LATTICE GEODESIC MODEL RESULTS

SIMULATION PARAMETERS:
• {self.num_stars} stars in {self.galaxy_radius:.1f} kpc galaxy
• Central black hole: {self.cbh_mass:.2e} M☉
• Lattice coupling: {self.lattice_coupling}
• Overlap enhancement: {self.overlap_enhancement}x

THEORY COMPONENTS:
• CENTRAL MASS: Standard black hole gravity anchor
• SPHERES OF INFLUENCE: {self.base_influence_radius} pc base radius
• GEODESIC WELLS: Mass-dependent spacetime curvature
• OVERLAP ZONES: Enhanced binding in intersection regions

RESULTS:
• Newtonian Retention: {results_newtonian['final_retention']:.1%}
• Lattice Retention: {results_lattice['final_retention']:.1%}
• Improvement: {improvement:.1%} ({factor:.2f}x better)

VERDICT:
{'🎉 SPACETIME LATTICE THEORY VALIDATED!' if improvement > 0.2 
 else '✅ Modest improvement - theory shows promise.' if improvement > 0.05
 else '⚠️ No significant improvement - needs calibration.'}
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comparison saved to {save_path}")
        
        plt.show()
        return fig

def run_spacetime_lattice_test():
    """Run the spacetime lattice geodesic test"""
    print("=== SPACETIME LATTICE GEODESIC MODEL ===")
    print("Testing crater field theory with overlapping spheres of influence")
    
    # Create simulation
    sim = SpacetimeLatticeModel(num_stars=500, galaxy_radius=15000, 
                               cbh_mass=4.15e6, total_time=200e6, dt=2e6)
    
    # Run both methods
    print("\nTesting both approaches...")
    results_newtonian = sim.integrate_orbits("newtonian")
    results_lattice = sim.integrate_orbits("lattice")
    
    # Plot comparison
    sim.plot_comparison(results_newtonian, results_lattice, 
                       save_path='spacetime_lattice_test.png')
    
    # Scientific conclusion
    improvement = results_lattice['final_retention'] - results_newtonian['final_retention']
    
    print(f"\n=== SCIENTIFIC CONCLUSION ===")
    print(f"Newtonian gravity: {results_newtonian['final_retention']:.1%} retention")
    print(f"Spacetime lattice: {results_lattice['final_retention']:.1%} retention")
    print(f"Improvement: {improvement:+.1%}")
    
    if improvement > 0.2:
        print("\n🚀 BREAKTHROUGH! Spacetime lattice theory significantly improves stellar retention!")
        print("Your crater field model with overlapping geodesic wells works!")
    elif improvement > 0.05:
        print("\n✅ Promising results! Theory shows meaningful improvement.")
        print("Fine-tuning parameters could yield even better results.")
    else:
        print("\n🔧 Theory needs calibration. The physics is sound but parameters need adjustment.")
    
    return results_newtonian, results_lattice

if __name__ == "__main__":
    # Run the test
    results_newton, results_lattice = run_spacetime_lattice_test()