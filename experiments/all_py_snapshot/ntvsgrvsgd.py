# Three-Way Gravity Test: Newtonian vs General Relativity vs Geodesic Theory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import time

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled!")
except ImportError:
    print("GPU not available, using CPU...")
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy

class ThreeGravityTest:
    def __init__(self, num_stars=1000, galaxy_radius=15000, cbh_mass=4.15e6, 
                 total_time=200e6, dt=2e6):
        """
        Compare three theories of gravity in galaxy dynamics:
        1. Newtonian (inverse square law)
        2. General Relativity (post-Newtonian corrections)
        3. Geodesic Theory (exponential kernel)
        """
        self.G = 4.30091e-6  # km/s, kpc, solar mass units
        self.c = 299792.458  # speed of light km/s
        self.num_stars = num_stars
        self.galaxy_radius = galaxy_radius / 1000  # Convert to kpc
        self.cbh_mass = cbh_mass  # Realistic Milky Way mass
        self.total_time = total_time
        self.dt = dt
        self.num_steps = int(total_time / dt)
        
        # Geodesic parameters (from your SPARC analysis)
        self.alpha = 0.905
        self.ell = 0.25 * self.galaxy_radius
        
        # Escape criteria
        self.escape_radius = 3.0 * self.galaxy_radius
        
        self._setup_initial_conditions()
        
    def _setup_initial_conditions(self):
        """Set up realistic stellar distribution"""
        print(f"Setting up {self.num_stars} stars in {self.galaxy_radius:.1f} kpc galaxy...")
        print(f"Central black hole: {self.cbh_mass:.2e} solar masses")
        
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
        
        # Realistic stellar masses
        self.star_masses = np.random.uniform(0.5, 3.0, self.num_stars)
        
        # Initial velocities (circular orbits for Newtonian case)
        v_circular = np.sqrt(self.G * self.cbh_mass / self.r)
        v_dispersion = 0.1 * v_circular
        v_radial = np.random.normal(0, v_dispersion, self.num_stars)
        v_tangential = v_circular + np.random.normal(0, v_dispersion, self.num_stars)
        
        # Convert to Cartesian velocities (on GPU if available)
        if GPU_AVAILABLE:
            self.vx = cp.array(v_radial * np.cos(angles) - v_tangential * np.sin(angles))
            self.vy = cp.array(v_radial * np.sin(angles) + v_tangential * np.cos(angles))
            self.x = cp.array(self.x)
            self.y = cp.array(self.y)
            self.r = cp.array(self.r)
            self.star_masses = cp.array(self.star_masses)
        else:
            self.vx = v_radial * np.cos(angles) - v_tangential * np.sin(angles)
            self.vy = v_radial * np.sin(angles) + v_tangential * np.cos(angles)
        
        print(f"Initial setup complete: stars from 0.5 to {self.galaxy_radius:.1f} kpc")
    
    def calculate_newtonian_acceleration(self, x, y):
        """Pure Newtonian gravity (inverse square law) - GPU accelerated"""
        if GPU_AVAILABLE:
            r = cp.sqrt(x**2 + y**2 + 0.01**2)  # Softening parameter
            a_magnitude = self.G * self.cbh_mass / r**2
            ax = -a_magnitude * x / r
            ay = -a_magnitude * y / r
        else:
            r = np.sqrt(x**2 + y**2 + 0.01**2)
            a_magnitude = self.G * self.cbh_mass / r**2
            ax = -a_magnitude * x / r
            ay = -a_magnitude * y / r
        
        return ax, ay
    
    def calculate_gr_acceleration(self, x, y, vx, vy):
        """General Relativity (1st post-Newtonian corrections) - GPU accelerated"""
        if GPU_AVAILABLE:
            r = cp.sqrt(x**2 + y**2 + 0.01**2)
            v2 = vx**2 + vy**2
            
            # Newtonian term
            a_newt = self.G * self.cbh_mass / r**2
            ax_newt = -a_newt * x / r
            ay_newt = -a_newt * y / r
            
            # Post-Newtonian corrections
            rs = 2 * self.G * self.cbh_mass / self.c**2  # Schwarzschild radius
            
            # Main 1PN corrections
            pn_factor = 1 + (3 * v2 / self.c**2) + (4 * self.G * self.cbh_mass / (r * self.c**2))
            
            # Frame dragging (Lense-Thirring effect) - simplified
            lt_factor = 1 + (rs / r) * 0.1  # Simplified approximation
            
            ax_gr = ax_newt * pn_factor * lt_factor
            ay_gr = ay_newt * pn_factor * lt_factor
        else:
            r = np.sqrt(x**2 + y**2 + 0.01**2)
            v2 = vx**2 + vy**2
            
            a_newt = self.G * self.cbh_mass / r**2
            ax_newt = -a_newt * x / r
            ay_newt = -a_newt * y / r
            
            rs = 2 * self.G * self.cbh_mass / self.c**2
            pn_factor = 1 + (3 * v2 / self.c**2) + (4 * self.G * self.cbh_mass / (r * self.c**2))
            lt_factor = 1 + (rs / r) * 0.1
            
            ax_gr = ax_newt * pn_factor * lt_factor
            ay_gr = ay_newt * pn_factor * lt_factor
        
        return ax_gr, ay_gr
    
    def calculate_geodesic_acceleration(self, x, y):
        """Your geodesic spacetime theory (exponential kernel) - GPU accelerated"""
        if GPU_AVAILABLE:
            r = cp.sqrt(x**2 + y**2 + 0.01**2)
            
            # Initialize acceleration arrays on GPU
            ax = cp.zeros_like(x)
            ay = cp.zeros_like(y)
            
            # Central black hole provides base gravitational well
            cbh_factor = 0.1  # Reduced central contribution
            a_central = cbh_factor * self.G * self.cbh_mass / r**2
            ax += -a_central * x / r
            ay += -a_central * y / r
            
            # Vectorized geodesic coupling (GPU optimized)
            # Create distance matrices
            x_diff = x[:, cp.newaxis] - x[cp.newaxis, :]
            y_diff = y[:, cp.newaxis] - y[cp.newaxis, :]
            dr_matrix = cp.sqrt(x_diff**2 + y_diff**2 + 0.01**2)
            
            # Radial distance matrix
            r_matrix = cp.sqrt(x**2 + y**2 + 0.01**2)
            r_diff_matrix = cp.abs(r_matrix[:, cp.newaxis] - r_matrix[cp.newaxis, :])
            
            # Exponential kernel matrix
            kernel_matrix = cp.exp(-r_diff_matrix / self.ell)
            
            # Mass matrix
            mass_matrix = self.star_masses[cp.newaxis, :] * cp.ones((len(x), len(x)))
            
            # Force calculation (vectorized)
            force_magnitude = (self.alpha * self.G * mass_matrix * kernel_matrix / 
                             (dr_matrix**2 * self.num_stars + 1e-10))
            
            # Set diagonal to zero (no self-interaction)
            cp.fill_diagonal(force_magnitude, 0)
            
            # Calculate accelerations
            ax_contrib = cp.sum(force_magnitude * x_diff / dr_matrix, axis=1)
            ay_contrib = cp.sum(force_magnitude * y_diff / dr_matrix, axis=1)
            
            ax -= ax_contrib
            ay -= ay_contrib
            
        else:
            # CPU fallback (slower but still works)
            r = np.sqrt(x**2 + y**2 + 0.01**2)
            ax = np.zeros_like(x)
            ay = np.zeros_like(y)
            
            cbh_factor = 0.1
            a_central = cbh_factor * self.G * self.cbh_mass / r**2
            ax += -a_central * x / r
            ay += -a_central * y / r
            
            # Simplified geodesic coupling (reduced N for CPU)
            step_size = max(1, len(x) // 100)  # Sample subset for CPU
            for i in range(0, len(x), step_size):
                for j in range(0, len(x), step_size):
                    if i != j:
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dr = np.sqrt(dx**2 + dy**2 + 0.01**2)
                        
                        r_i = np.sqrt(x[i]**2 + y[i]**2 + 0.01**2)
                        r_j = np.sqrt(x[j]**2 + y[j]**2 + 0.01**2)
                        
                        kernel_weight = np.exp(-abs(r_i - r_j) / self.ell)
                        force_magnitude = (self.alpha * self.G * self.star_masses[j] * 
                                         kernel_weight / (dr**2 * self.num_stars))
                        
                        ax[i] -= force_magnitude * dx / dr
                        ay[i] -= force_magnitude * dy / dr
        
        return ax, ay
    
    def integrate_orbits(self, gravity_type="newtonian"):
        """Integrate stellar orbits using specified gravity theory"""
        theories = {
            "newtonian": "Newtonian (Inverse Square)",
            "gr": "General Relativity (1PN)",
            "geodesic": "Geodesic Spacetime Theory"
        }
        
        print(f"\n=== {theories[gravity_type]} ===")
        
        # Copy initial conditions (ensure GPU arrays)
        if GPU_AVAILABLE:
            x, y = cp.array(self.x), cp.array(self.y)
            vx, vy = cp.array(self.vx), cp.array(self.vy)
        else:
            x, y = self.x.copy(), self.y.copy()
            vx, vy = self.vx.copy(), self.vy.copy()
        
        # Tracking arrays
        bound_mask = cp.ones(self.num_stars, dtype=bool) if GPU_AVAILABLE else np.ones(self.num_stars, dtype=bool)
        
        # Snapshot storage
        snapshot_interval = int(25e6 / self.dt)  # Every 25 Myr
        snapshots = []
        times = []
        
        start_time = time.time()
        
        for step in range(self.num_steps):
            # Calculate accelerations based on theory
            if gravity_type == "newtonian":
                ax, ay = self.calculate_newtonian_acceleration(x, y)
            elif gravity_type == "gr":
                ax, ay = self.calculate_gr_acceleration(x, y, vx, vy)
            elif gravity_type == "geodesic":
                ax, ay = self.calculate_geodesic_acceleration(x, y)
            
            # Leapfrog integration
            dt_code = self.dt * 3.154e7 / 3.086e13  # Convert to code units
            
            # Update velocities
            vx += ax * dt_code
            vy += ay * dt_code
            
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
            
            # Store snapshots (transfer to CPU for storage)
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
            'theory': theories[gravity_type],
            'snapshots': snapshots,
            'times': times,
            'final_retention': final_retention,
            'final_bound': final_bound
        }
    
    def plot_three_way_comparison(self, results_newton, results_gr, results_geodesic, save_path=None):
        """Plot comprehensive three-way comparison"""
        fig = plt.figure(figsize=(18, 12))
        
        # Get final snapshots
        snap_n = results_newton['snapshots'][-1] if results_newton['snapshots'] else {'x': [], 'y': [], 'bound_count': 0}
        snap_gr = results_gr['snapshots'][-1] if results_gr['snapshots'] else {'x': [], 'y': [], 'bound_count': 0}
        snap_geo = results_geodesic['snapshots'][-1] if results_geodesic['snapshots'] else {'x': [], 'y': [], 'bound_count': 0}
        
        # 1. Final galaxy states (top row)
        colors = ['red', 'blue', 'gold']
        titles = ['Newtonian Gravity', 'General Relativity', 'Geodesic Theory']
        snaps = [snap_n, snap_gr, snap_geo]
        
        for i, (snap, color, title) in enumerate(zip(snaps, colors, titles)):
            ax = plt.subplot(3, 3, i+1)
            
            if len(snap['x']) > 0:
                ax.scatter(snap['x'], snap['y'], s=1, alpha=0.7, color=color)
            
            # Galaxy boundary
            circle = patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                                  edgecolor='white', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
            
            ax.set_xlim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
            ax.set_ylim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
            title_text = f'{title}\n{snap["bound_count"]} stars retained'
            ax.set_title(title_text, fontsize=12, color='white')
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.set_aspect('equal')
        
        # 2. Retention over time (middle left)
        ax4 = plt.subplot(3, 3, 4)
        
        if results_newton['snapshots']:
            newton_retention = [s['bound_count']/self.num_stars for s in results_newton['snapshots']]
            ax4.plot(results_newton['times'], newton_retention, 'red', linewidth=2, label='Newtonian')
        
        if results_gr['snapshots']:
            gr_retention = [s['bound_count']/self.num_stars for s in results_gr['snapshots']]
            ax4.plot(results_gr['times'], gr_retention, 'blue', linewidth=2, label='General Relativity')
        
        if results_geodesic['snapshots']:
            geo_retention = [s['bound_count']/self.num_stars for s in results_geodesic['snapshots']]
            ax4.plot(results_geodesic['times'], geo_retention, 'gold', linewidth=2, label='Geodesic Theory')
        
        ax4.set_xlabel('Time (Myr)')
        ax4.set_ylabel('Fraction Retained')
        ax4.set_title('Stellar Retention Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        # 3. Radial density profiles (middle center)
        ax5 = plt.subplot(3, 3, 5)
        r_bins = np.linspace(0, self.galaxy_radius, 15)
        
        for snap, color, label in zip(snaps, colors, ['Newtonian', 'GR', 'Geodesic']):
            if len(snap.get('r', [])) > 0:
                hist, _ = np.histogram(snap['r'], bins=r_bins)
                r_centers = (r_bins[:-1] + r_bins[1:]) / 2
                ax5.plot(r_centers, hist, color=color, linewidth=2, label=label)
        
        ax5.set_xlabel('Radius (kpc)')
        ax5.set_ylabel('Number of Stars')
        ax5.set_title('Final Radial Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 4. Final retention comparison (middle right)
        ax6 = plt.subplot(3, 3, 6)
        theories_names = ['Newtonian', 'General\nRelativity', 'Geodesic\nTheory']
        retentions = [
            results_newton['final_retention'],
            results_gr['final_retention'], 
            results_geodesic['final_retention']
        ]
        
        bars = ax6.bar(theories_names, retentions, color=colors, alpha=0.8)
        ax6.set_ylabel('Final Retention Rate')
        ax6.set_title('Theory Comparison')
        ax6.set_ylim(0, 1.0)
        
        # Add percentage labels on bars
        for bar, retention in zip(bars, retentions):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{retention:.1%}', ha='center', va='bottom', fontweight='bold')
        
        ax6.grid(True, alpha=0.3)
        
        # 5. Physics summary (bottom span)
        ax7 = plt.subplot(3, 1, 3)
        ax7.axis('off')
        
        verdict = self._generate_verdict(results_newton, results_gr, results_geodesic)
        
        summary_text = f"""
GALACTIC DYNAMICS: THREE THEORIES OF GRAVITY COMPARISON

SIMULATION PARAMETERS:
• {self.num_stars} stars in {self.galaxy_radius:.1f} kpc galaxy
• Central black hole: {self.cbh_mass:.2e} M☉ 
• Simulation time: {self.total_time/1e6:.0f} Myr
• Geodesic parameters: α = {self.alpha}, ℓ = {self.ell:.2f} kpc

THEORY DESCRIPTIONS:
• NEWTONIAN: Pure inverse-square law gravity (F ∝ 1/r²)
• GENERAL RELATIVITY: Post-Newtonian corrections + frame dragging
• GEODESIC THEORY: Exponential kernel coupling (exp(-Δr/ℓ))

RESULTS:
• Newtonian Retention: {results_newton['final_retention']:.1%}
• GR Retention: {results_gr['final_retention']:.1%}  
• Geodesic Retention: {results_geodesic['final_retention']:.1%}

VERDICT:
{verdict}
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Three-way comparison saved to {save_path}")
        
        plt.show()
        return fig
    
    def _generate_verdict(self, results_n, results_gr, results_geo):
        """Generate scientific verdict based on results"""
        retentions = [results_n['final_retention'], results_gr['final_retention'], results_geo['final_retention']]
        best_idx = np.argmax(retentions)
        theories = ['Newtonian', 'General Relativity', 'Geodesic Theory']
        
        best_theory = theories[best_idx]
        best_retention = retentions[best_idx]
        
        if best_retention > 0.8:
            verdict = f"{best_theory} provides excellent stellar retention ({best_retention:.1%})."
        elif best_retention > 0.5:
            verdict = f"{best_theory} shows promising results ({best_retention:.1%} retention)."
        else:
            verdict = "All theories show significant stellar loss. Galaxy formation remains challenging."
        
        # Special cases
        if results_geo['final_retention'] > max(results_n['final_retention'], results_gr['final_retention']) + 0.1:
            verdict += "\n🎉 GEODESIC THEORY VALIDATED! Outperforms both Newtonian and GR approaches."
        elif abs(results_geo['final_retention'] - results_gr['final_retention']) < 0.05:
            verdict += "\n✅ Geodesic theory matches General Relativity predictions!"
        
        return verdict

def run_three_gravity_test():
    """Run the complete three-way gravity comparison"""
    print("=== THREE-WAY GRAVITY COMPARISON TEST ===")
    print("Comparing Newtonian, General Relativity, and Geodesic Spacetime theories")
    
    # Create simulation
    sim = ThreeGravityTest(num_stars=1000, galaxy_radius=15000, 
                          cbh_mass=4.15e6, total_time=200e6, dt=2e6)
    
    # Run all three theories
    print("\nTesting all three theories of gravity...")
    results_newton = sim.integrate_orbits("newtonian")
    results_gr = sim.integrate_orbits("gr") 
    results_geodesic = sim.integrate_orbits("geodesic")
    
    # Plot comprehensive comparison
    sim.plot_three_way_comparison(results_newton, results_gr, results_geodesic,
                                 save_path='three_gravity_comparison.png')
    
    # Print scientific conclusion
    print(f"\n=== SCIENTIFIC CONCLUSION ===")
    print(f"Newtonian gravity: {results_newton['final_retention']:.1%} retention")
    print(f"General Relativity: {results_gr['final_retention']:.1%} retention")  
    print(f"Geodesic Theory: {results_geodesic['final_retention']:.1%} retention")
    
    # Determine which theories agree
    if abs(results_geodesic['final_retention'] - results_gr['final_retention']) < 0.05:
        print("\n🔬 GEODESIC THEORY MATCHES GENERAL RELATIVITY!")
        print("Your spacetime geometry approach reproduces Einstein's predictions.")
    
    if results_geodesic['final_retention'] > results_newton['final_retention'] + 0.1:
        print("\n🚀 GEODESIC THEORY SUPERIOR TO NEWTONIAN GRAVITY!")
        print("Exponential kernel provides better galactic binding than inverse-square law.")
    
    return results_newton, results_gr, results_geodesic

if __name__ == "__main__":
    # Run the definitive test
    run_three_gravity_test()