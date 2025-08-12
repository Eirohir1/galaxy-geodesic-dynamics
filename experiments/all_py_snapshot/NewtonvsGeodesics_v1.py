# GPU-Accelerated Realistic Galaxy Model with Full Structure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
from scipy.special import gammaincinv
import time

# GPU acceleration setup
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU ACCELERATION ENABLED!")
    # Get GPU info
    gpu_device = cp.cuda.Device()
    memory_info = gpu_device.mem_info
    print("GPU Memory: {:.1f} GB available".format(memory_info[0] / 1e9))
except ImportError:
    print("⚠️ CuPy not found - using CPU (will be SLOW!)")
    print("Install CuPy for GPU acceleration: pip install cupy-cuda11x")
    GPU_AVAILABLE = False
    cp = np

class RealisticGalaxy:
    """
    GPU-accelerated realistic galaxy with:
    - Central black hole (CBH)
    - Dense bulge/core
    - Exponential disk
    - Spiral arm structure
    - Proper velocity distribution
    """
    
    def __init__(self, 
                 num_stars=10000,
                 cbh_mass=4.15e6,  # Sgr A* mass
                 bulge_mass=1e10,   # Central bulge mass
                 disk_mass=6e10,    # Disk mass
                 galaxy_radius=20,  # kpc
                 scale_height=0.3,  # kpc (disk thickness)
                 num_arms=2,        # Number of spiral arms
                 arm_pitch=25,      # Spiral arm pitch angle (degrees)
                 bar_strength=0.3,  # Bar strength (0-1)
                 use_gpu=True):     # Force GPU usage
        
        self.G = 4.30091e-6  # km²/s², kpc, solar mass units
        self.num_stars = num_stars
        self.cbh_mass = cbh_mass
        self.bulge_mass = bulge_mass
        self.disk_mass = disk_mass
        self.galaxy_radius = galaxy_radius
        self.scale_height = scale_height
        self.num_arms = num_arms
        self.arm_pitch = np.radians(arm_pitch)
        self.bar_strength = bar_strength
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        # Component scales
        self.bulge_radius = galaxy_radius * 0.2  # Bulge is inner 20%
        self.disk_scale_length = galaxy_radius / 3.5  # Standard disk scale
        
        # Initialize galaxy
        self._create_galaxy_structure()
        
    def _create_galaxy_structure(self):
        """Build the full galaxy with all components"""
        print("\n" + "="*60)
        print("BUILDING GPU-ACCELERATED GALAXY MODEL")
        print("="*60)
        
        # Determine stellar populations
        n_bulge = int(0.25 * self.num_stars)  # 25% in bulge
        n_disk = int(0.60 * self.num_stars)   # 60% in disk
        n_arms = self.num_stars - n_bulge - n_disk  # 15% in spiral arms
        
        print("Stellar distribution:")
        print("  Central bulge: " + str(n_bulge) + " stars")
        print("  Disk: " + str(n_disk) + " stars")
        print("  Spiral arms: " + str(n_arms) + " stars")
        print("  Using GPU: " + str(self.use_gpu))
        
        # 1. CREATE CENTRAL BULGE
        bulge_positions, bulge_velocities, bulge_masses = self._create_bulge(n_bulge)
        
        # 2. CREATE EXPONENTIAL DISK
        disk_positions, disk_velocities, disk_masses = self._create_disk(n_disk)
        
        # 3. CREATE SPIRAL ARMS
        arm_positions, arm_velocities, arm_masses = self._create_spiral_arms(n_arms)
        
        # Combine all components (CPU first for construction)
        positions_cpu = np.vstack([bulge_positions, disk_positions, arm_positions])
        velocities_cpu = np.vstack([bulge_velocities, disk_velocities, arm_velocities])
        masses_cpu = np.hstack([bulge_masses, disk_masses, arm_masses])
        
        # Convert to GPU arrays if available
        if self.use_gpu:
            print("\nTransferring to GPU...")
            self.positions = cp.asarray(positions_cpu)
            self.velocities = cp.asarray(velocities_cpu)
            self.masses = cp.asarray(masses_cpu)
            
            self.x = self.positions[:, 0]
            self.y = self.positions[:, 1]
            self.z = self.positions[:, 2]
            self.vx = self.velocities[:, 0]
            self.vy = self.velocities[:, 1]
            self.vz = self.velocities[:, 2]
        else:
            self.positions = positions_cpu
            self.velocities = velocities_cpu
            self.masses = masses_cpu
            
            self.x = positions_cpu[:, 0]
            self.y = positions_cpu[:, 1]
            self.z = positions_cpu[:, 2]
            self.vx = velocities_cpu[:, 0]
            self.vy = velocities_cpu[:, 1]
            self.vz = velocities_cpu[:, 2]
        
        # Component labels (keep on CPU for now)
        self.component_labels = np.array(['bulge']*n_bulge + ['disk']*n_disk + ['arms']*n_arms)
        
        self._calculate_statistics()
        
    def _create_bulge(self, n_stars):
        """Create dense central bulge with proper kinematics"""
        print("\nCreating central bulge...")
        
        # Hernquist profile for bulge
        a = self.bulge_radius * 0.5  # Scale radius
        
        # Generate radii using inverse transform sampling
        u = np.random.uniform(0, 1, n_stars)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))
        r = np.clip(r, 0, self.bulge_radius * 2)
        
        # 3D positions (spherical bulge)
        theta = np.random.uniform(0, 2*np.pi, n_stars)
        phi = np.arccos(1 - 2*np.random.uniform(0, 1, n_stars))
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) * 0.6  # Slightly flattened
        
        positions = np.column_stack([x, y, z])
        
        # Stellar masses - more massive stars in bulge
        masses = np.random.lognormal(0.3, 0.5, n_stars)  # Log-normal distribution
        masses = np.clip(masses, 0.3, 10.0)  # Limit range
        
        # Velocities - combination of rotation and dispersion
        r_cylindrical = np.sqrt(x**2 + y**2)
        
        # Rotation curve in bulge
        v_rot = self._rotation_curve(r_cylindrical, component='bulge')
        
        # High velocity dispersion in bulge
        sigma_bulge = 100  # km/s
        v_disp_r = np.random.normal(0, sigma_bulge, n_stars)
        v_disp_theta = np.random.normal(0, sigma_bulge, n_stars)
        v_disp_z = np.random.normal(0, sigma_bulge * 0.7, n_stars)
        
        # Combine rotation and dispersion
        v_theta = v_rot + v_disp_theta
        
        # Convert to Cartesian
        vx = v_disp_r * np.cos(theta) - v_theta * np.sin(theta)
        vy = v_disp_r * np.sin(theta) + v_theta * np.cos(theta)
        vz = v_disp_z
        
        velocities = np.column_stack([vx, vy, vz])
        
        return positions, velocities, masses
    
    def _create_disk(self, n_stars):
        """Create exponential disk with proper rotation"""
        print("Creating exponential disk...")
        
        # Exponential radial distribution
        r = np.random.exponential(self.disk_scale_length, n_stars)
        r = np.clip(r, self.bulge_radius, self.galaxy_radius)
        
        # Uniform angular distribution (before spiral modulation)
        theta = np.random.uniform(0, 2*np.pi, n_stars)
        
        # Vertical distribution (sech² profile)
        z = self.scale_height * np.arctanh(np.random.uniform(-0.99, 0.99, n_stars))
        
        # Add bar potential if specified
        if self.bar_strength > 0:
            # Simple bar model - elliptical distortion
            bar_angle = np.pi/4  # Bar orientation
            r_bar = r * (1 + self.bar_strength * np.cos(2*(theta - bar_angle)))
            x = r_bar * np.cos(theta)
            y = r_bar * np.sin(theta)
        else:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        
        positions = np.column_stack([x, y, z])
        
        # Stellar masses - standard IMF for disk
        masses = self._sample_imf(n_stars)
        
        # Velocities - circular orbits with asymmetric drift
        v_circ = self._rotation_curve(r, component='disk')
        
        # Add velocity dispersion (less than bulge)
        sigma_r = 30 * np.exp(-r / (2 * self.disk_scale_length))  # Decreases outward
        sigma_theta = sigma_r * 0.7  # Epicyclic ratio
        sigma_z = sigma_r * 0.5
        
        v_r = np.random.normal(0, sigma_r, n_stars)
        v_theta = v_circ + np.random.normal(0, sigma_theta, n_stars)
        v_z = np.random.normal(0, sigma_z, n_stars)
        
        # Convert to Cartesian
        vx = v_r * np.cos(theta) - v_theta * np.sin(theta)
        vy = v_r * np.sin(theta) + v_theta * np.cos(theta)
        vz = v_z
        
        velocities = np.column_stack([vx, vy, vz])
        
        return positions, velocities, masses
    
    def _create_spiral_arms(self, n_stars):
        """Create logarithmic spiral arm structure"""
        print("Creating spiral arms...")
        
        # Start with exponential disk distribution
        r = np.random.exponential(self.disk_scale_length * 1.2, n_stars)
        r = np.clip(r, self.bulge_radius * 1.5, self.galaxy_radius)
        
        # Create spiral pattern
        arm_assignment = np.random.randint(0, self.num_arms, n_stars)
        
        theta_base = []
        for i in range(n_stars):
            arm_number = arm_assignment[i]
            
            # Logarithmic spiral: r = a * exp(b * theta)
            # Rearranged: theta = log(r/a) / b
            b = np.tan(self.arm_pitch)
            a = self.bulge_radius
            
            # Base spiral angle
            theta_spiral = np.log(r[i] / a) / b
            
            # Add arm offset
            theta_offset = 2 * np.pi * arm_number / self.num_arms
            
            # Add some spread around the arm
            theta_spread = np.random.normal(0, 0.1)  # radians
            
            theta = theta_spiral + theta_offset + theta_spread
            theta_base.append(theta)
        
        theta = np.array(theta_base)
        
        # Positions
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = self.scale_height * np.random.normal(0, 0.5, n_stars)  # Thin disk
        
        positions = np.column_stack([x, y, z])
        
        # Stellar masses - younger population in arms
        masses = self._sample_imf(n_stars, young=True)
        
        # Velocities - similar to disk but with streaming motions
        v_circ = self._rotation_curve(r, component='disk')
        
        # Spiral arm streaming motion
        v_stream = 10  # km/s streaming velocity
        
        # Velocity components
        v_r = np.random.normal(v_stream, 20, n_stars)  # Outward streaming
        v_theta = v_circ + np.random.normal(0, 25, n_stars)
        v_z = np.random.normal(0, 15, n_stars)
        
        # Convert to Cartesian
        vx = v_r * np.cos(theta) - v_theta * np.sin(theta)
        vy = v_r * np.sin(theta) + v_theta * np.cos(theta)
        vz = v_z
        
        velocities = np.column_stack([vx, vy, vz])
        
        return positions, velocities, masses
    
    def _rotation_curve(self, r, component='disk'):
        """
        Calculate rotation curve including all mass components
        This creates the flat rotation curve observed in real galaxies
        """
        # CBH contribution
        v_cbh = np.sqrt(self.G * self.cbh_mass / (r + 0.001))  # Softening
        
        # Bulge contribution (Hernquist profile)
        a = self.bulge_radius * 0.5
        m_bulge_enclosed = self.bulge_mass * r**2 / (r + a)**2
        v_bulge = np.sqrt(self.G * m_bulge_enclosed / r)
        
        # Disk contribution (exponential)
        m_disk_enclosed = self.disk_mass * (1 - (1 + r/self.disk_scale_length) * 
                                           np.exp(-r/self.disk_scale_length))
        v_disk = np.sqrt(self.G * m_disk_enclosed / (r + 0.1))
        
        # Combine in quadrature
        v_total = np.sqrt(v_cbh**2 + v_bulge**2 + v_disk**2)
        
        # Add "dark matter" effect for now (to be replaced by geodesics)
        v_flat = 220  # km/s - typical galaxy rotation speed
        v_total = np.sqrt(v_total**2 + v_flat**2 * (1 - np.exp(-r / self.disk_scale_length)))
        
        return v_total
    
    def _sample_imf(self, n_stars, young=False):
        """Sample stellar masses from Initial Mass Function"""
        if young:
            # Younger population - more massive stars
            masses = np.random.lognormal(0.5, 0.7, n_stars)
            masses = np.clip(masses, 0.5, 50.0)
        else:
            # Standard Kroupa IMF approximation
            masses = []
            for _ in range(n_stars):
                u = np.random.uniform()
                if u < 0.5:
                    m = np.random.uniform(0.08, 0.5)
                elif u < 0.9:
                    m = np.random.uniform(0.5, 1.0)
                else:
                    m = np.random.uniform(1.0, 8.0)
                masses.append(m)
            masses = np.array(masses)
        
        return masses
    
    def _calculate_statistics(self):
        """Calculate and display galaxy statistics"""
        print("\n" + "="*60)
        print("GALAXY STATISTICS")
        print("="*60)
        
        # Get CPU arrays for statistics
        if self.use_gpu:
            x_cpu = cp.asnumpy(self.x)
            y_cpu = cp.asnumpy(self.y)
            z_cpu = cp.asnumpy(self.z)
            masses_cpu = cp.asnumpy(self.masses)
            vx_cpu = cp.asnumpy(self.vx)
            vy_cpu = cp.asnumpy(self.vy)
            vz_cpu = cp.asnumpy(self.vz)
        else:
            x_cpu = self.x
            y_cpu = self.y
            z_cpu = self.z
            masses_cpu = self.masses
            vx_cpu = self.vx
            vy_cpu = self.vy
            vz_cpu = self.vz
        
        # Radial distribution
        r = np.sqrt(x_cpu**2 + y_cpu**2)
        
        print("\nMass distribution:")
        print("  Total stellar mass: {:.2e} M☉".format(np.sum(masses_cpu)))
        print("  Average stellar mass: {:.2f} M☉".format(np.mean(masses_cpu)))
        print("  Mass range: {:.2f} - {:.2f} M☉".format(np.min(masses_cpu), np.max(masses_cpu)))
        
        print("\nSpatial distribution:")
        print("  Mean radius: {:.2f} kpc".format(np.mean(r)))
        print("  Core (<2 kpc): {} stars".format(np.sum(r < 2)))
        print("  Disk (2-15 kpc): {} stars".format(np.sum((r >= 2) & (r < 15))))
        print("  Periphery (>15 kpc): {} stars".format(np.sum(r >= 15)))
        
        # Velocity statistics
        v_total = np.sqrt(vx_cpu**2 + vy_cpu**2 + vz_cpu**2)
        print("\nKinematics:")
        print("  Mean velocity: {:.1f} km/s".format(np.mean(v_total)))
        print("  Velocity dispersion: {:.1f} km/s".format(np.std(v_total)))
        
        # Density profile
        density_core = np.sum(r < 2) / (np.pi * 4)  # stars per kpc²
        density_disk = np.sum((r >= 2) & (r < 15)) / (np.pi * (225 - 4))
        print("\nDensity:")
        print("  Core density: {:.1f} stars/kpc²".format(density_core))
        print("  Disk density: {:.1f} stars/kpc²".format(density_disk))
    
    def simulate_orbits_gpu(self, time_myr=100, dt_myr=1):
        """
        GPU-accelerated orbital simulation with proper galaxy potential
        """
        if not self.use_gpu:
            print("WARNING: GPU not available, simulation will be slow!")
        
        print("\n" + "="*60)
        print("GPU-ACCELERATED ORBITAL SIMULATION")
        print("="*60)
        
        n_steps = int(time_myr / dt_myr)
        dt_code = dt_myr * 1e6 * 3.154e7 / 3.086e13  # Convert to code units
        
        print("Simulating {} Myr with {} steps...".format(time_myr, n_steps))
        start_time = time.time()
        
        initial_radius = cp.sqrt(self.x**2 + self.y**2) if self.use_gpu else np.sqrt(self.x**2 + self.y**2)
        
        for step in range(n_steps):
            if self.use_gpu:
                # GPU calculation
                r_cyl = cp.sqrt(self.x**2 + self.y**2 + 0.001**2)
                r_sph = cp.sqrt(self.x**2 + self.y**2 + self.z**2 + 0.001**2)
                
                # Total galaxy potential (CBH + bulge + disk)
                # Using rotation curve to get total acceleration
                v_rot_sq = self._rotation_curve_gpu(r_cyl)**2
                a_radial = v_rot_sq / r_cyl
                
                # Project to x,y,z
                ax = -a_radial * self.x / r_cyl
                ay = -a_radial * self.y / r_cyl
                az = -self.G * self.cbh_mass * self.z / r_sph**3  # Vertical restoring force
                
                # Update velocities and positions
                self.vx += ax * dt_code
                self.vy += ay * dt_code
                self.vz += az * dt_code
                
                self.x += self.vx * dt_code
                self.y += self.vy * dt_code
                self.z += self.vz * dt_code
            else:
                # CPU fallback
                r_cyl = np.sqrt(self.x**2 + self.y**2 + 0.001**2)
                r_sph = np.sqrt(self.x**2 + self.y**2 + self.z**2 + 0.001**2)
                
                # Use rotation curve for radial acceleration
                v_rot_sq = self._rotation_curve(r_cyl)**2
                a_radial = v_rot_sq / r_cyl
                
                self.vx -= a_radial * self.x / r_cyl * dt_code
                self.vy -= a_radial * self.y / r_cyl * dt_code
                self.vz -= self.G * self.cbh_mass * self.z / r_sph**3 * dt_code
                
                self.x += self.vx * dt_code
                self.y += self.vy * dt_code
                self.z += self.vz * dt_code
            
            if step % max(1, n_steps // 10) == 0:
                if self.use_gpu:
                    r_current = cp.sqrt(self.x**2 + self.y**2)
                    escaped = cp.sum(r_current > self.galaxy_radius * 3)
                else:
                    r_current = np.sqrt(self.x**2 + self.y**2)
                    escaped = np.sum(r_current > self.galaxy_radius * 3)
                
                progress = 100.0 * (step + 1) / n_steps
                print("\r  Progress: {:.1f}%, Escaped: {}".format(progress, int(escaped)), end="", flush=True)
        
        print()  # New line after progress
        elapsed = time.time() - start_time
        print("\nSimulation complete in {:.1f} seconds".format(elapsed))
        if elapsed > 0.001:
            print("Speed: {:.0f} star-steps per second".format(self.num_stars * n_steps / elapsed))
        else:
            print("Speed: Too fast to measure accurately")
        
        if self.use_gpu:
            print("GPU Memory used: {:.1f} MB".format(
                (self.positions.nbytes + self.velocities.nbytes + self.masses.nbytes) / 1e6))
    
    def _rotation_curve_gpu(self, r):
        """GPU version of rotation curve"""
        if self.use_gpu:
            # CBH contribution
            v_cbh = cp.sqrt(self.G * self.cbh_mass / (r + 0.001))
            
            # Bulge contribution
            a = self.bulge_radius * 0.5
            m_bulge_enclosed = self.bulge_mass * r**2 / (r + a)**2
            v_bulge = cp.sqrt(self.G * m_bulge_enclosed / r)
            
            # Disk contribution
            m_disk_enclosed = self.disk_mass * (1 - (1 + r/self.disk_scale_length) * 
                                               cp.exp(-r/self.disk_scale_length))
            v_disk = cp.sqrt(self.G * m_disk_enclosed / (r + 0.1))
            
            # Combine in quadrature
            v_total = cp.sqrt(v_cbh**2 + v_bulge**2 + v_disk**2)
            
            # Add flat component
            v_flat = 220
            v_total = cp.sqrt(v_total**2 + v_flat**2 * (1 - cp.exp(-r / self.disk_scale_length)))
            
            return v_total
        else:
            return self._rotation_curve(r)
    
    def plot_galaxy(self):
        """Visualize the galaxy structure"""
        # Get CPU arrays for plotting
        if self.use_gpu:
            x_plot = cp.asnumpy(self.x)
            y_plot = cp.asnumpy(self.y)
            z_plot = cp.asnumpy(self.z)
            masses_plot = cp.asnumpy(self.masses)
            vx_plot = cp.asnumpy(self.vx)
            vy_plot = cp.asnumpy(self.vy)
            vz_plot = cp.asnumpy(self.vz)
        else:
            x_plot = self.x
            y_plot = self.y
            z_plot = self.z
            masses_plot = self.masses
            vx_plot = self.vx
            vy_plot = self.vy
            vz_plot = self.vz
        
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Face-on view (X-Y)
        ax1 = plt.subplot(2, 3, 1)
        r = np.sqrt(x_plot**2 + y_plot**2)
        
        # Color by component
        colors = {'bulge': 'red', 'disk': 'blue', 'arms': 'gold'}
        for component in ['bulge', 'disk', 'arms']:
            mask = self.component_labels == component
            ax1.scatter(x_plot[mask], y_plot[mask], 
                       c=colors[component], s=1, alpha=0.5, label=component)
        
        ax1.set_xlim(-self.galaxy_radius, self.galaxy_radius)
        ax1.set_ylim(-self.galaxy_radius, self.galaxy_radius)
        ax1.set_xlabel('X (kpc)')
        ax1.set_ylabel('Y (kpc)')
        ax1.set_title('Face-on View')
        ax1.legend()
        ax1.set_aspect('equal')
        
        # Add circles for reference
        for radius in [2, 5, 10, 15, 20]:
            circle = Circle((0, 0), radius, fill=False, 
                          edgecolor='gray', linestyle='--', alpha=0.3)
            ax1.add_patch(circle)
        
        # 2. Edge-on view (X-Z)
        ax2 = plt.subplot(2, 3, 2)
        for component in ['bulge', 'disk', 'arms']:
            mask = self.component_labels == component
            ax2.scatter(x_plot[mask], z_plot[mask], 
                       c=colors[component], s=1, alpha=0.5)
        
        ax2.set_xlim(-self.galaxy_radius, self.galaxy_radius)
        ax2.set_ylim(-2, 2)
        ax2.set_xlabel('X (kpc)')
        ax2.set_ylabel('Z (kpc)')
        ax2.set_title('Edge-on View')
        ax2.set_aspect('equal')
        
        # 3. Rotation curve
        ax3 = plt.subplot(2, 3, 3)
        r_samples = np.linspace(0.1, self.galaxy_radius, 100)
        v_rot = self._rotation_curve(r_samples)
        ax3.plot(r_samples, v_rot, 'b-', linewidth=2, label='Model')
        
        # Plot actual stellar velocities
        r_stars = np.sqrt(x_plot**2 + y_plot**2)
        v_circ_stars = np.sqrt(vx_plot**2 + vy_plot**2)
        
        # Bin the data
        r_bins = np.linspace(0, self.galaxy_radius, 20)
        v_binned = []
        r_centers = []
        for i in range(len(r_bins)-1):
            mask = (r_stars >= r_bins[i]) & (r_stars < r_bins[i+1])
            if np.sum(mask) > 5:
                v_binned.append(np.mean(v_circ_stars[mask]))
                r_centers.append((r_bins[i] + r_bins[i+1])/2)
        
        ax3.scatter(r_centers, v_binned, c='red', s=30, alpha=0.7, label='Stars')
        ax3.set_xlabel('Radius (kpc)')
        ax3.set_ylabel('Rotation Velocity (km/s)')
        ax3.set_title('Rotation Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Density profile
        ax4 = plt.subplot(2, 3, 4)
        r_bins = np.linspace(0, self.galaxy_radius, 30)
        hist, _ = np.histogram(r, bins=r_bins)
        
        # Convert to surface density
        areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
        surface_density = hist / areas
        
        ax4.semilogy((r_bins[:-1] + r_bins[1:])/2, surface_density, 'b-', linewidth=2)
        ax4.set_xlabel('Radius (kpc)')
        ax4.set_ylabel('Surface Density (stars/kpc²)')
        ax4.set_title('Radial Density Profile')
        ax4.grid(True, alpha=0.3)
        
        # 5. Mass distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(masses_plot, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.set_xlabel('Stellar Mass (M☉)')
        ax5.set_ylabel('Number of Stars')
        ax5.set_title('Stellar Mass Distribution')
        ax5.set_yscale('log')
        
        # 6. GPU Performance info
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        info_text = """GPU ACCELERATION STATUS

"""
        if self.use_gpu:
            gpu_device = cp.cuda.Device()
            memory_info = gpu_device.mem_info
            data_size = (self.positions.nbytes + self.velocities.nbytes + self.masses.nbytes) / 1e6
            
            info_text += """✅ GPU ENABLED
Memory Available: {:.1f} GB
Memory Used: {:.1f} MB

Performance Estimate:
• {} stars
• ~1000x faster than CPU
• Can handle ~1M stars

Ready for geodesic simulation!""".format(
                memory_info[0] / 1e9,
                data_size,
                self.num_stars
            )
        else:
            info_text += """⚠️ GPU NOT AVAILABLE
Running on CPU (slow)

To enable GPU:
1. Install CUDA
2. pip install cupy-cuda11x
3. Restart script

Performance limited to
~10k stars on CPU"""
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', 
                         facecolor='lightgreen' if self.use_gpu else 'yellow', 
                         alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('gpu_galaxy_structure.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_for_simulation(self):
        """Export galaxy data for geodesic simulations"""
        # Convert to CPU arrays for export
        if self.use_gpu:
            positions_export = cp.asnumpy(self.positions)
            velocities_export = cp.asnumpy(self.velocities)
            masses_export = cp.asnumpy(self.masses)
        else:
            positions_export = self.positions
            velocities_export = self.velocities
            masses_export = self.masses
            
        return {
            'positions': positions_export,
            'velocities': velocities_export,
            'masses': masses_export,
            'components': self.component_labels,
            'parameters': {
                'cbh_mass': self.cbh_mass,
                'bulge_mass': self.bulge_mass,
                'disk_mass': self.disk_mass,
                'galaxy_radius': self.galaxy_radius,
                'num_stars': self.num_stars
            },
            'gpu_ready': self.use_gpu
        }

def create_milky_way_model(num_stars=10000):
    """Create a GPU-accelerated Milky Way-like galaxy"""
    print("="*60)
    print("CREATING GPU-ACCELERATED MILKY WAY MODEL")
    print("="*60)
    
    galaxy = RealisticGalaxy(
        num_stars=num_stars,
        cbh_mass=4.15e6,    # Sagittarius A*
        bulge_mass=2e10,    # Milky Way bulge
        disk_mass=6e10,     # Milky Way disk
        galaxy_radius=20,    # kpc
        scale_height=0.3,    # Thin disk
        num_arms=4,          # Four spiral arms
        arm_pitch=12,        # Tighter spiral
        bar_strength=0.3,    # Milky Way has a bar
        use_gpu=True         # Force GPU usage
    )
    
    return galaxy

def benchmark_gpu_performance():
    """Test GPU vs CPU performance"""
    print("\n" + "="*60)
    print("GPU PERFORMANCE BENCHMARK")
    print("="*60)
    
    test_sizes = [1000, 5000, 10000, 50000]
    
    for n_stars in test_sizes:
        print("\nTesting with {} stars:".format(n_stars))
        
        # GPU test
        if GPU_AVAILABLE:
            print("  GPU: ", end="")
            start = time.time()
            galaxy_gpu = RealisticGalaxy(num_stars=n_stars, use_gpu=True)
            galaxy_gpu.simulate_orbits_gpu(time_myr=10, dt_myr=0.5)
            gpu_time = time.time() - start
            print("    {:.2f} seconds".format(gpu_time))
        else:
            print("  GPU: Not available")
            gpu_time = None
        
        # CPU test (only for small sizes)
        if n_stars <= 5000:
            print("  CPU: ", end="")
            start = time.time()
            galaxy_cpu = RealisticGalaxy(num_stars=n_stars, use_gpu=False)
            galaxy_cpu.simulate_orbits_gpu(time_myr=10, dt_myr=0.5)
            cpu_time = time.time() - start
            print("    {:.2f} seconds".format(cpu_time))
            
            if gpu_time:
                speedup = cpu_time / gpu_time
                print("  Speedup: {:.1f}x faster on GPU".format(speedup))
        else:
            print("  CPU: Skipped (too slow)")

if __name__ == "__main__":
    # Check GPU status first
    if GPU_AVAILABLE:
        print("🚀 GPU ACCELERATION AVAILABLE!")
        # Run benchmark
        benchmark_gpu_performance()
    else:
        print("⚠️ WARNING: GPU not available - install CuPy for acceleration!")
        print("Continuing with CPU (will be slow)...")
    
    print("\n" + "="*60)
    
    # Create the galaxy
    galaxy = create_milky_way_model(num_stars=10000 if GPU_AVAILABLE else 1000)
    
    # Visualize structure
    galaxy.plot_galaxy()
    
    # Quick orbital test
    if GPU_AVAILABLE:
        galaxy.simulate_orbits_gpu(time_myr=50, dt_myr=1)
    
    # Export for geodesic testing
    galaxy_data = galaxy.export_for_simulation()
    
    print("\n" + "="*60)
    print("GALAXY MODEL COMPLETE")
    print("="*60)
    print("Ready for geodesic raindrop simulation!")
    print("Exported data contains:")
    print("  - {} stellar positions".format(len(galaxy_data['positions'])))
    print("  - {} stellar velocities".format(len(galaxy_data['velocities'])))
    print("  - {} stellar masses".format(len(galaxy_data['masses'])))
    print("  - GPU Ready: {}".format(galaxy_data['gpu_ready']))
    print("\nNext step: Run geodesic escape test with this realistic structure!")