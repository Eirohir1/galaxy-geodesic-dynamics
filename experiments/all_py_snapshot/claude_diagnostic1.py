# Scientific Gravity Theory Diagnostic
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class GravityDiagnostic:
    def __init__(self, num_test_stars=50):
        """
        Systematic diagnostic of gravity theories
        Starting with minimal complexity to isolate issues
        """
        self.G = 4.30091e-6  # km/s, kpc, solar mass units
        self.c = 299792.458  # speed of light km/s
        
        # Test galaxy parameters (small for debugging)
        self.cbh_mass = 4.15e6  # Realistic Milky Way mass
        self.galaxy_radius = 10.0  # 10 kpc test galaxy
        self.num_test_stars = num_test_stars
        
        # Original SPARC parameters
        self.alpha_sparc = 0.905
        self.ell_sparc = 0.25 * self.galaxy_radius
        
        print("=== GRAVITY THEORY DIAGNOSTIC ===")
        print(f"Testing with {num_test_stars} stars in {self.galaxy_radius} kpc galaxy")
        print(f"CBH mass: {self.cbh_mass:.2e} M☉")
        print(f"SPARC parameters: α = {self.alpha_sparc}, ℓ = {self.ell_sparc:.2f} kpc")
    
    def setup_test_orbits(self):
        """Create carefully controlled test orbits"""
        # Place test stars at specific radii for analysis
        radii = np.linspace(1.0, self.galaxy_radius, self.num_test_stars)
        angles = np.linspace(0, 2*np.pi, self.num_test_stars)
        
        self.x = radii * np.cos(angles)
        self.y = radii * np.sin(angles)
        self.r = radii
        
        # All stars have same mass for simplicity
        self.star_masses = np.ones(self.num_test_stars)
        
        print(f"Test orbits: {len(radii)} stars from 1.0 to {self.galaxy_radius:.1f} kpc")
        
        return self.x, self.y, self.r
    
    def test_newtonian_force(self, x, y):
        """Test pure Newtonian gravity"""
        r = np.sqrt(x**2 + y**2 + 0.001**2)
        
        # Force magnitude
        F_mag = self.G * self.cbh_mass / r**2
        
        # Force components (attractive toward center)
        Fx = -F_mag * x / r
        Fy = -F_mag * y / r
        
        return Fx, Fy, F_mag
    
    def test_geodesic_force(self, x, y, debug=True):
        """Test geodesic force with detailed diagnostics"""
        r = np.sqrt(x**2 + y**2 + 0.001**2)
        
        # Initialize forces
        Fx_geo = np.zeros_like(x)
        Fy_geo = np.zeros_like(y)
        
        if debug:
            print(f"\\nGeodесic force calculation:")
            print(f"ℓ = {self.ell_sparc:.3f} kpc")
            print(f"α = {self.alpha_sparc:.3f}")
        
        # Pairwise geodesic interactions
        force_matrix = np.zeros((len(x), len(x)))
        
        for i in range(len(x)):
            for j in range(len(x)):
                if i != j:
                    # Distance between stars
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dr = np.sqrt(dx**2 + dy**2 + 0.001**2)
                    
                    # Radial positions
                    r_i = r[i]
                    r_j = r[j]
                    
                    # Exponential kernel
                    kernel = np.exp(-abs(r_i - r_j) / self.ell_sparc)
                    
                    # Force magnitude (per your theory)
                    F_magnitude = (self.alpha_sparc * self.G * self.star_masses[j] * 
                                 kernel / dr**2)
                    
                    force_matrix[i,j] = F_magnitude
                    
                    # Force components (attractive: toward other star)
                    Fx_geo[i] -= F_magnitude * dx / dr
                    Fy_geo[i] -= F_magnitude * dy / dr
        
        if debug and len(x) <= 10:
            print(f"Force matrix (first few):")
            print(force_matrix[:min(5,len(x)), :min(5,len(x))])
        
        return Fx_geo, Fy_geo, force_matrix
    
    def test_circular_velocity(self, r_test):
        """Calculate circular velocities for different theories"""
        results = {}
        
        # Newtonian circular velocity
        v_circ_newton = np.sqrt(self.G * self.cbh_mass / r_test)
        results['newtonian'] = v_circ_newton
        
        # For geodesic theory, we need to solve the equilibrium
        # This is complex, so let's approximate
        x_test = r_test
        y_test = np.zeros_like(r_test)
        
        # Get forces
        Fx_newt, Fy_newt, F_newt = self.test_newtonian_force(x_test, y_test)
        
        # For geodesic force, we need background stars
        # Use a simplified approach with fixed background
        self.setup_test_orbits()
        
        # Calculate geodesic forces for each test radius
        Fx_geo_total = np.zeros_like(r_test)
        Fy_geo_total = np.zeros_like(r_test)
        
        for i, (x_pos, y_pos) in enumerate(zip(x_test, y_test)):
            # Single test position
            x_single = np.array([x_pos])
            y_single = np.array([y_pos])
            
            # Combine with background
            x_combined = np.concatenate([x_single, self.x])
            y_combined = np.concatenate([y_single, self.y])
            
            # Create combined mass array
            masses_combined = np.concatenate([np.array([1.0]), self.star_masses])
            
            # Temporarily update masses
            original_masses = self.star_masses
            self.star_masses = masses_combined
            
            # Calculate geodesic force
            Fx_geo, Fy_geo, _ = self.test_geodesic_force(x_combined, y_combined, debug=False)
            
            # Restore masses
            self.star_masses = original_masses
            
            # Store force on test star (first element)
            Fx_geo_total[i] = Fx_geo[0]
            Fy_geo_total[i] = Fy_geo[0]
        
        # Total radial force magnitude
        F_total = np.sqrt((Fx_newt + Fx_geo_total)**2 + (Fy_newt + Fy_geo_total)**2)
        
        # Circular velocity for combined forces
        v_circ_combined = np.sqrt(F_total * r_test)
        results['geodesic'] = v_circ_combined
        
        return results
    
    def compare_force_profiles(self):
        """Compare force profiles vs radius"""
        radii = np.linspace(0.5, self.galaxy_radius, 50)
        
        # Set up test positions along x-axis
        x_test = radii
        y_test = np.zeros_like(radii)
        
        # Calculate forces
        Fx_newt, Fy_newt, F_newt = self.test_newtonian_force(x_test, y_test)
        
        # For geodesic, we need a representative stellar distribution
        self.setup_test_orbits()
        
        # Test geodesic forces at each radius
        F_geo_profile = []
        for i, r in enumerate(radii):
            # Single test star at radius r
            x_single = np.array([r])
            y_single = np.array([0.0])
            
            # Add background stellar distribution
            x_full = np.concatenate([x_single, self.x])
            y_full = np.concatenate([y_single, self.y])
            
            # Create masses array for full system (test star + background)
            masses_full = np.concatenate([np.array([1.0]), self.star_masses])
            
            # Temporarily store original masses and update
            original_masses = self.star_masses
            self.star_masses = masses_full
            
            Fx_geo, Fy_geo, _ = self.test_geodesic_force(x_full, y_full, debug=False)
            
            # Restore original masses
            self.star_masses = original_masses
            
            # Extract force on test star (first element)
            F_geo_total = np.sqrt(Fx_geo[0]**2 + Fy_geo[0]**2)
            F_geo_profile.append(F_geo_total)
        
        F_geo_profile = np.array(F_geo_profile)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        # Force profiles
        plt.subplot(2, 2, 1)
        plt.loglog(radii, F_newt, 'r-', label='Newtonian', linewidth=2)
        plt.loglog(radii, F_geo_profile, 'g-', label='Geodesic', linewidth=2)
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Force Magnitude')
        plt.title('Gravitational Force vs Radius')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Circular velocity curves
        plt.subplot(2, 2, 2)
        v_curves = self.test_circular_velocity(radii)
        for theory, v_curve in v_curves.items():
            plt.plot(radii, v_curve, linewidth=2, label=theory.capitalize())
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Circular Velocity (km/s)')
        plt.title('Rotation Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Force ratio
        plt.subplot(2, 2, 3)
        ratio = F_geo_profile / F_newt
        plt.semilogx(radii, ratio, 'b-', linewidth=2)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Radius (kpc)')
        plt.ylabel('F_geodesic / F_newtonian')
        plt.title('Force Ratio')
        plt.grid(True, alpha=0.3)
        
        # Stellar distribution
        plt.subplot(2, 2, 4)
        plt.scatter(self.x, self.y, s=20, alpha=0.7, color='blue')
        circle = patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                              edgecolor='red', linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.xlim(-self.galaxy_radius*1.1, self.galaxy_radius*1.1)
        plt.ylim(-self.galaxy_radius*1.1, self.galaxy_radius*1.1)
        plt.xlabel('x (kpc)')
        plt.ylabel('y (kpc)')
        plt.title('Test Stellar Distribution')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('gravity_diagnostic.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'radii': radii,
            'F_newtonian': F_newt,
            'F_geodesic': F_geo_profile,
            'v_curves': v_curves
        }
    
    def test_single_orbit(self, r_init=5.0, steps=1000, dt=1e6):
        """Test a single star orbit under different theories"""
        print(f"\\nTesting single orbit at r = {r_init} kpc")
        
        # Initial conditions
        x_init = r_init
        y_init = 0.0
        
        # Calculate appropriate circular velocity for each theory
        v_curves = self.test_circular_velocity(np.array([r_init]))
        
        results = {}
        
        for theory in ['newtonian', 'geodesic']:
            print(f"\\nTesting {theory} orbit...")
            
            # Initial velocity (circular)
            v_circ = v_curves[theory][0]
            vx_init = 0.0
            vy_init = v_circ
            
            print(f"Initial conditions: r = {r_init:.1f} kpc, v = {v_circ:.1f} km/s")
            
            # Integrate orbit
            x, y = [x_init], [y_init]
            vx, vy = [vx_init], [vy_init]
            
            x_curr, y_curr = x_init, y_init
            vx_curr, vy_curr = vx_init, vy_init
            
            for step in range(steps):
                # Calculate forces
                if theory == 'newtonian':
                    Fx, Fy, _ = self.test_newtonian_force(np.array([x_curr]), np.array([y_curr]))
                    ax, ay = Fx[0], Fy[0]
                else:  # geodesic
                    # Need background stars for geodesic calculation
                    self.setup_test_orbits()
                    x_full = np.concatenate([[x_curr], self.x])
                    y_full = np.concatenate([[y_curr], self.y])
                    
                    # Create masses array for full system
                    masses_full = np.concatenate([np.array([1.0]), self.star_masses])
                    original_masses = self.star_masses
                    self.star_masses = masses_full
                    
                    # Newtonian + geodesic
                    Fx_n, Fy_n, _ = self.test_newtonian_force(np.array([x_curr]), np.array([y_curr]))
                    Fx_g, Fy_g, _ = self.test_geodesic_force(x_full, y_full, debug=False)
                    
                    # Restore masses
                    self.star_masses = original_masses
                    
                    ax = Fx_n[0] + Fx_g[0]
                    ay = Fy_n[0] + Fy_g[0]
                
                # Leapfrog integration
                dt_code = dt * 3.154e7 / 3.086e13
                
                vx_curr += ax * dt_code
                vy_curr += ay * dt_code
                
                x_curr += vx_curr * dt_code
                y_curr += vy_curr * dt_code
                
                # Store every 10th step
                if step % 10 == 0:
                    x.append(x_curr)
                    y.append(y_curr)
                    vx.append(vx_curr)
                    vy.append(vy_curr)
                
                # Check for escape
                r_curr = np.sqrt(x_curr**2 + y_curr**2)
                if r_curr > 3 * self.galaxy_radius:
                    print(f"  Star escaped at step {step}, r = {r_curr:.1f} kpc")
                    break
            
            results[theory] = {
                'x': np.array(x),
                'y': np.array(y),
                'vx': np.array(vx),
                'vy': np.array(vy),
                'escaped': r_curr > 3 * self.galaxy_radius,
                'final_r': r_curr
            }
            
            print(f"  Final: r = {r_curr:.2f} kpc, {'ESCAPED' if results[theory]['escaped'] else 'BOUND'}")
        
        # Plot orbits
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        for theory, data in results.items():
            color = 'red' if theory == 'newtonian' else 'green'
            plt.plot(data['x'], data['y'], color=color, linewidth=2, label=theory.capitalize())
        
        # Starting point
        plt.plot(x_init, y_init, 'ko', markersize=8, label='Start')
        
        # Galaxy boundary
        circle = patches.Circle((0, 0), self.galaxy_radius, fill=False, 
                              edgecolor='blue', linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
        
        plt.xlim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
        plt.ylim(-self.galaxy_radius*1.2, self.galaxy_radius*1.2)
        plt.xlabel('x (kpc)')
        plt.ylabel('y (kpc)')
        plt.title('Orbital Trajectories')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Radius vs time
        plt.subplot(1, 2, 2)
        for theory, data in results.items():
            r_time = np.sqrt(data['x']**2 + data['y']**2)
            times = np.arange(len(r_time)) * dt * 10 / 1e6  # Myr
            color = 'red' if theory == 'newtonian' else 'green'
            plt.plot(times, r_time, color=color, linewidth=2, label=theory.capitalize())
        
        plt.axhline(y=r_init, color='k', linestyle=':', alpha=0.7, label='Initial radius')
        plt.xlabel('Time (Myr)')
        plt.ylabel('Radius (kpc)')
        plt.title('Radial Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('single_orbit_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results
    
    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("\\n" + "="*50)
        print("RUNNING FULL GRAVITY DIAGNOSTIC")
        print("="*50)
        
        # Test 1: Set up test configuration
        print("\\n1. Setting up test stellar distribution...")
        self.setup_test_orbits()
        
        # Test 2: Compare force profiles
        print("\\n2. Analyzing force profiles...")
        force_data = self.compare_force_profiles()
        
        # Test 3: Single orbit test
        print("\\n3. Testing single star orbits...")
        orbit_data = self.test_single_orbit()
        
        # Summary analysis
        print("\\n" + "="*50)
        print("DIAGNOSTIC SUMMARY")
        print("="*50)
        
        # Check force magnitudes
        F_ratio_median = np.median(force_data['F_geodesic'] / force_data['F_newtonian'])
        print(f"Median F_geodesic/F_newtonian ratio: {F_ratio_median:.3f}")
        
        if F_ratio_median > 10:
            print("⚠️  WARNING: Geodesic forces are much stronger than Newtonian")
        elif F_ratio_median < 0.1:
            print("⚠️  WARNING: Geodesic forces are much weaker than Newtonian")
        else:
            print("✓ Geodesic and Newtonian forces are comparable")
        
        # Check orbit stability
        newtonian_stable = not orbit_data['newtonian']['escaped']
        geodesic_stable = not orbit_data['geodesic']['escaped']
        
        print(f"Single orbit test:")
        print(f"  Newtonian: {'STABLE' if newtonian_stable else 'ESCAPED'}")
        print(f"  Geodesic:  {'STABLE' if geodesic_stable else 'ESCAPED'}")
        
        if newtonian_stable and not geodesic_stable:
            print("🔍 DIAGNOSIS: Geodesic forces are destabilizing orbits")
        elif not newtonian_stable and geodesic_stable:
            print("✨ RESULT: Geodesic forces stabilize orbits better than Newtonian")
        elif newtonian_stable and geodesic_stable:
            print("✓ Both theories produce stable orbits")
        else:
            print("⚠️  Both theories produce unstable orbits - check parameters")
        
        return {
            'force_data': force_data,
            'orbit_data': orbit_data,
            'summary': {
                'force_ratio': F_ratio_median,
                'newtonian_stable': newtonian_stable,
                'geodesic_stable': geodesic_stable
            }
        }

if __name__ == "__main__":
    # Run diagnostic
    diagnostic = GravityDiagnostic(num_test_stars=20)
    results = diagnostic.run_full_diagnostic()