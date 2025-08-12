# Galactic Stellar Retention Test: Standard vs Geodesic
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import cupy as cp
import time

class GalacticRetentionTest:
    def __init__(self, num_stars=2000, galaxy_radius=15000, cbh_mass=1e10, 
                 total_time=500e6, dt=1e6):  # 500 Myr simulation, 1 Myr steps
        """
        Test stellar retention in standard vs geodesic gravity
        
        Parameters:
        - num_stars: Number of test stars
        - galaxy_radius: Maximum initial radius (parsecs)
        - cbh_mass: Central black hole mass (solar masses)
        - total_time: Total simulation time (years)
        - dt: Time step (years)
        """
        self.G = 4.30091e-6  # km/s, kpc, solar mass units
        self.num_stars = num_stars
        self.galaxy_radius = galaxy_radius / 1000  # Convert to kpc
        self.cbh_mass = cbh_mass
        self.total_time = total_time
        self.dt = dt
        self.num_steps = int(total_time / dt)
        
        # Physics parameters
        self.escape_factor = 2.0  # v_escape = sqrt(2) * v_circular
        
        # Geodesic parameters (from your SPARC analysis)
        self.alpha = 0.905  # Geodesic coupling strength
        self.ell_factor = 0.25  # ℓ = 0.25 * R_galaxy
        
        self._setup_initial_conditions()
        
    def _setup_initial_conditions(self):
        """Set up realistic initial stellar orbits"""
        print(f"Setting up {self.num_stars} stars in {self.galaxy_radius:.1f} kpc galaxy...")
        
        # Generate stellar positions (disk distribution)
        np.random.seed(42)
        
        # Exponential disk profile
        scale_length = self.galaxy_radius / 3
        radii = np.random.exponential(scale_length, self.num_stars)
        radii = np.clip(radii, 0.5, self.galaxy_radius)  # Min 0.5 kpc from center
        
        angles = np.random.uniform(0, 2*np.pi, self.num_stars)
        
        # Convert to Cartesian coordinates
        self.x = radii * np.cos(angles)
        self.y = radii * np.sin(angles)
        self.r = radii
        
        # Initial velocities for circular orbits (standard gravity only)
        v_circular = np.sqrt(self.G * self.cbh_mass / self.r)
        
        # Add velocity perturbations (realistic galaxy)
        v_dispersion = 0.1 * v_circular  # 10% velocity dispersion
        v_radial = np.random.normal(0, v_dispersion, self.num_stars)
        v_tangential = v_circular + np.random.normal(0, v_dispersion, self.num_stars)
        
        # Convert to Cartesian velocities
        self.vx = v_radial * np.cos(angles) - v_tangential * np.sin(angles)
        self.vy = v_radial * np.sin(angles) + v_tangential * np.cos(angles)
        
        # Star masses (for visualization)
        self.masses = np.random.uniform(0.5, 3.0, self.num_stars)
        
        print(f"Initial setup: {len(self.r)} stars from 0.5 to {self.galaxy_radius:.1f} kpc")
        
    def calculate_acceleration_standard(self, x, y):
        """Calculate acceleration using standard Newtonian gravity only"""
        r = np.sqrt(x**2 + y**2)
        
        # Central black hole gravity
        a_magnitude = self.G * self.cbh_mass / (r**2 + 0.01**2)  # Softening
        ax = -a_magnitude * x / r
        ay = -a_magnitude * y / r
        
        return ax, ay
    
    def calculate_acceleration_geodesic(self, x, y):
        """Calculate acceleration with geodesic enhancement"""
        # Start with standard gravity
        ax_std, ay_std = self.calculate_acceleration_standard(x, y)
        
        # Geodesic enhancement using exponential kernel
        r = np.sqrt(x**2 + y**2)
        ell = self.ell_factor * self.galaxy_radius
        
        # Create geodesic coupling matrix
        ax_geo = np.zeros_like(ax_std)
        ay_geo = np.zeros_like(ay_std)
        
        for i in range(len(x)):
            # Calculate geodesic influence from all other masses
            r_i = r[i]
            
            # Exponential kernel: influence falls as exp(-|r-r'|/ℓ)
            for j in range(len(x)):
                if i != j:
                    r_j = r[j]
                    dr = abs(r_i - r_j)
                    kernel_weight = np.exp(-dr / ell)
                    
                    # Add geodesic coupling (simplified)
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr_vec = np.sqrt(dx**2 + dy**2) + 0.01
                    
                    geo_strength = self.alpha * kernel_weight / (self.num_stars * dr_vec**2)
                    ax_geo[i] -= geo_strength * dx / dr_vec
                    ay_geo[i] -= geo_strength * dy / dr_vec
        
        return ax_std + ax_geo, ay_std + ay_geo
    
    def integrate_orbits(self, use_geodesics=False):
        """Integrate stellar orbits using leapfrog method"""
        method_name = "Geodesic" if use_geodesics else "Standard"
        print(f"\n=== {method_name} Gravity Simulation ===")
        
        # Copy initial conditions
        x, y = self.x.copy(), self.y.copy()
        vx, vy = self.vx.copy(), self.vy.copy()
        
        # Tracking arrays
        ejected_mask = np.zeros(self.num_stars, dtype=bool)
        ejection_times = np.full(self.num_stars, -1)
        
        # Snapshot storage (every 50 Myr)
        snapshot_interval = int(50e6 / self.dt)
        snapshots = []
        times = []
        
        start_time = time.time()
        
        for step in range(self.num_steps):
            # Calculate accelerations
            if use_geodesics:
                ax, ay = self.calculate_acceleration_geodesic(x, y)
            else:
                ax, ay = self.calculate_acceleration_standard(x, y)
            
            # Leapfrog integration
            dt_years = self.dt
            dt_code = dt_years * 3.154e7 / 3.086e13  # Convert to code units
            
            # Update velocities
            vx += ax * dt_code
            vy += ay * dt_code
            
            # Update positions
            x += vx * dt_code
            y += vy * dt_code
            
            # Check for ejections
            r_current = np.sqrt(x**2 + y**2)
            v_current = np.sqrt(vx**2 + vy**2)
            v_escape = np.sqrt(2 * self.G * self.cbh_mass / r_current)
            
            # Mark newly ejected stars
            newly_ejected = (v_current > self.escape_factor * v_escape) & (~ejected_mask)
            ejected_mask |= newly_ejected
            ejection_times[newly_ejected] = step * self.dt / 1e6  # Store in Myr
            
            # Store snapshots
            if step % snapshot_interval == 0:
                bound_mask = ~ejected_mask & (r_current < 2 * self.galaxy_radius)
                snapshots.append({
                    'x': x[bound_mask].copy(),
                    'y': y[bound_mask].copy(),
                    'r': r_current[bound_mask].copy(),
                    'v': v_current[bound_mask].copy(),
                    'bound_count': np.sum(bound_mask),
                    'ejected_count': np.sum(ejected_mask)
                })
                times.append(step * self.dt / 1e6)  # Store in Myr
                
                if step % (10 * snapshot_interval) == 0:
                    bound_frac = np.sum(bound_mask) / self.num_stars
                    print(f"  t={step * self.dt / 1e6:.0f} Myr: {bound_frac:.1%} stars bound")
        
        elapsed = time.time() - start_time
        final_bound = np.sum(~ejected_mask & (r_current < 2 * self.galaxy_radius))
        final_ejected = np.sum(ejected_mask)
        
        print(f"  Simulation completed in {elapsed:.1f}s")
        print(f"  Final: {final_bound} bound, {final_ejected} ejected")
        print(f"  Retention rate: {final_bound/self.num_stars:.1%}")
        
        return {
            'method': method_name,
            'snapshots': snapshots,
            'times': times,
            'ejected_mask': ejected_mask,
            'ejection_times': ejection_times,
            'final_bound': final_bound,
            'final_ejected': final_ejected,
            'retention_rate': final_bound / self.num_stars
        }
    
    def plot_comparison(self, results_standard, results_geodesic, save_path=None):
        """Plot side-by-side comparison of both methods"""
        fig = plt.figure(figsize=(16, 12))
        
        # Get final snapshots
        snap_std = results_standard['snapshots'][-1]
        snap_geo = results_geodesic['snapshots'][-1]
        
        # 1. Final galaxy states
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(snap_std['x'], snap_std['y'], s=1, alpha=0.6, color='cyan')
        ax1.add_patch(patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                                   edgecolor='white', linestyle='--', alpha=0.5))
        ax1.set_xlim(-self.galaxy_radius*1.5, self.galaxy_radius*1.5)
        ax1.set_ylim(-self.galaxy_radius*1.5, self.galaxy_radius*1.5)
        title1 = f'Standard Gravity\n{snap_std["bound_count"]} stars retained'
        ax1.set_title(title1, color='white', fontsize=12)
        ax1.set_facecolor('black')
        ax1.tick_params(colors='white')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.scatter(snap_geo['x'], snap_geo['y'], s=1, alpha=0.6, color='yellow')
        ax2.add_patch(patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                                   edgecolor='white', linestyle='--', alpha=0.5))
        ax2.set_xlim(-self.galaxy_radius*1.5, self.galaxy_radius*1.5)
        ax2.set_ylim(-self.galaxy_radius*1.5, self.galaxy_radius*1.5)
        title2 = f'Geodesic Enhanced\n{snap_geo["bound_count"]} stars retained'
        ax2.set_title(title2, color='white', fontsize=12)
        ax2.set_facecolor('black')
        ax2.tick_params(colors='white')
        
        # 2. Retention over time
        ax3 = plt.subplot(2, 3, 3)
        times = results_standard['times']
        std_bound = [s['bound_count']/self.num_stars for s in results_standard['snapshots']]
        geo_bound = [s['bound_count']/self.num_stars for s in results_geodesic['snapshots']]
        
        ax3.plot(times, std_bound, 'cyan', linewidth=2, label='Standard Gravity')
        ax3.plot(times, geo_bound, 'yellow', linewidth=2, label='Geodesic Enhanced')
        ax3.set_xlabel('Time (Myr)')
        ax3.set_ylabel('Fraction of Stars Retained')
        ax3.set_title('Stellar Retention Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        
        # 3. Radial density profiles
        ax4 = plt.subplot(2, 3, 4)
        r_bins = np.linspace(0, self.galaxy_radius, 20)
        
        std_hist, _ = np.histogram(snap_std['r'], bins=r_bins)
        geo_hist, _ = np.histogram(snap_geo['r'], bins=r_bins)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        
        ax4.plot(r_centers, std_hist, 'cyan', linewidth=2, label='Standard')
        ax4.plot(r_centers, geo_hist, 'yellow', linewidth=2, label='Geodesic')
        ax4.set_xlabel('Radius (kpc)')
        ax4.set_ylabel('Number of Stars')
        ax4.set_title('Final Radial Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 4. Ejection time histogram
        ax5 = plt.subplot(2, 3, 5)
        std_ejections = results_standard['ejection_times'][results_standard['ejection_times'] > 0]
        geo_ejections = results_geodesic['ejection_times'][results_geodesic['ejection_times'] > 0]
        
        if len(std_ejections) > 0:
            ax5.hist(std_ejections, bins=20, alpha=0.7, color='cyan', 
                    label=f'Standard ({len(std_ejections)} ejected)')
        if len(geo_ejections) > 0:
            ax5.hist(geo_ejections, bins=20, alpha=0.7, color='yellow', 
                    label=f'Geodesic ({len(geo_ejections)} ejected)')
        
        ax5.set_xlabel('Ejection Time (Myr)')
        ax5.set_ylabel('Number of Stars')
        ax5.set_title('When Stars Were Ejected')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 5. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
        GALACTIC RETENTION TEST RESULTS
        
        Initial Stars: {self.num_stars}
        Galaxy Radius: {self.galaxy_radius:.1f} kpc
        Simulation Time: {self.total_time/1e6:.0f} Myr
        
        STANDARD GRAVITY:
        • Stars Retained: {results_standard['final_bound']}
        • Stars Ejected: {results_standard['final_ejected']}
        • Retention Rate: {results_standard['retention_rate']:.1%}
        
        GEODESIC ENHANCED:
        • Stars Retained: {results_geodesic['final_bound']}
        • Stars Ejected: {results_geodesic['final_ejected']}
        • Retention Rate: {results_geodesic['retention_rate']:.1%}
        
        IMPROVEMENT:
        • Δ Retention: {results_geodesic['retention_rate'] - results_standard['retention_rate']:.1%}
        • Factor: {results_geodesic['retention_rate'] / results_standard['retention_rate']:.2f}x
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Comparison saved to {save_path}")
        
        plt.show()
        
        return fig

def run_retention_test():
    """Run the complete retention test"""
    print("=== GALACTIC STELLAR RETENTION TEST ===")
    print("Testing whether geodesic spacetime theory prevents stellar ejections")
    
    # Create simulation
    sim = GalacticRetentionTest(num_stars=2000, galaxy_radius=15000, 
                               total_time=500e6, dt=2e6)
    
    # Run both methods
    results_standard = sim.integrate_orbits(use_geodesics=False)
    results_geodesic = sim.integrate_orbits(use_geodesics=True)
    
    # Plot comparison
    sim.plot_comparison(results_standard, results_geodesic, 
                       save_path='galactic_retention_test.png')
    
    # Print final verdict
    improvement = results_geodesic['retention_rate'] - results_standard['retention_rate']
    factor = results_geodesic['retention_rate'] / results_standard['retention_rate']
    
    print(f"\n=== FINAL VERDICT ===")
    print(f"Standard gravity retains {results_standard['retention_rate']:.1%} of stars")
    print(f"Geodesic theory retains {results_geodesic['retention_rate']:.1%} of stars")
    print(f"Improvement: +{improvement:.1%} ({factor:.2f}x better)")
    
    if improvement > 0.1:  # 10% improvement
        print("🎉 GEODESIC THEORY VALIDATED! Significantly better stellar retention.")
    elif improvement > 0.05:  # 5% improvement  
        print("✅ Geodesic theory shows promise - modest but meaningful improvement.")
    else:
        print("❌ No significant improvement - theory needs refinement.")
    
    return results_standard, results_geodesic

if __name__ == "__main__":
    # Run the test
    results_std, results_geo = run_retention_test()