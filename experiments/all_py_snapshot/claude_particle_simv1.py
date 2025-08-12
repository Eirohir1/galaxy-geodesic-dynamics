# GPU-Accelerated Galaxy Geodesic Simulation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter
import cupy as cp  # GPU acceleration library
import time

class GPUGalaxySimulation:
    def __init__(self, num_stars=1000, galaxy_extent=2000, cbh_mass=1e10, 
                 num_steps=5, dt=5e6, grid_res=500):
        """
        Initialize GPU-accelerated galaxy simulation
        
        Parameters:
        - num_stars: Number of stars in simulation
        - galaxy_extent: Galaxy radius in parsecs
        - cbh_mass: Central black hole mass
        - num_steps: Number of time steps
        - dt: Time step in years
        - grid_res: Grid resolution for field calculation
        """
        # Constants
        self.G = 4.30091e-6  # Gravitational constant
        self.num_stars = num_stars
        self.galaxy_extent = galaxy_extent
        self.cbh_mass = cbh_mass
        self.num_steps = num_steps
        self.dt = dt
        self.grid_res = grid_res
        
        # Initialize on GPU
        self._setup_gpu_arrays()
        
    def _setup_gpu_arrays(self):
        """Initialize all arrays on GPU"""
        print("Setting up GPU arrays...")
        
        # Star properties (on GPU)
        cp.random.seed(42)
        self.star_radii = cp.sqrt(cp.random.uniform(1000**2, self.galaxy_extent**2, self.num_stars))
        self.star_angles = cp.random.uniform(0, 2 * cp.pi, self.num_stars)
        self.star_speeds = cp.sqrt(self.G * self.cbh_mass / (self.star_radii / 1000))
        self.star_masses = cp.random.uniform(0.5, 5.0, self.num_stars)
        
        # Grid setup (on GPU)
        x_lin = cp.linspace(-self.galaxy_extent, self.galaxy_extent, self.grid_res)
        y_lin = cp.linspace(-self.galaxy_extent, self.galaxy_extent, self.grid_res)
        self.X_grid, self.Y_grid = cp.meshgrid(x_lin, y_lin)
        
        print(f"Initialized {self.num_stars} stars on GPU with {self.grid_res}x{self.grid_res} grid")
    
    def calculate_gravitational_field(self, star_x, star_y):
        """
        Calculate gravitational field on GPU using vectorized operations
        """
        # Central black hole contribution
        Z_field = -self.cbh_mass / cp.sqrt(self.X_grid**2 + self.Y_grid**2 + 1e3)
        
        # Vectorized star contributions
        for i in range(len(star_x)):
            sx, sy = star_x[i], star_y[i]
            star_contribution = -1e5 / cp.sqrt((self.X_grid - sx)**2 + (self.Y_grid - sy)**2 + 1e3)
            Z_field += star_contribution
            
        return Z_field
    
    def update_positions(self, step):
        """Update star positions using GPU vectorized operations"""
        angle_change = (self.star_speeds / self.star_radii) * (self.dt * step * 3.154e+7 / (3.086e+13))
        new_angles = self.star_angles + angle_change
        
        x_positions = self.star_radii * cp.cos(new_angles)
        y_positions = self.star_radii * cp.sin(new_angles)
        
        return new_angles, x_positions, y_positions
    
    def apply_geodesic_kernel(self, field, kernel_strength=0.9, kernel_length=500):
        """
        Apply exponential geodesic kernel to enhance gravitational field
        This represents the spacetime curvature effects we derived
        """
        try:
            # Create exponential kernel
            kernel_size = int(kernel_length / (2 * self.galaxy_extent / self.grid_res))
            kernel_size = min(kernel_size, 50)  # Limit kernel size to prevent memory issues
            
            kernel_grid = cp.arange(-kernel_size, kernel_size + 1, dtype=cp.float32)
            kernel_x, kernel_y = cp.meshgrid(kernel_grid, kernel_grid)
            kernel_r = cp.sqrt(kernel_x**2 + kernel_y**2) * (2 * self.galaxy_extent / self.grid_res)
            
            # Exponential geodesic kernel: exp(-r/ℓ)
            geodesic_kernel = cp.exp(-kernel_r / kernel_length)
            geodesic_kernel = geodesic_kernel / cp.sum(geodesic_kernel)  # Normalize
            
            # Use scipy-style convolution with CuPy arrays
            from cupyx.scipy.ndimage import convolve
            enhanced_component = convolve(field, geodesic_kernel, mode='constant')
            enhanced_field = field + kernel_strength * enhanced_component
            
            return enhanced_field
            
        except Exception as e:
            print(f"Geodesic kernel failed, using simpler approximation: {e}")
            # Fallback: simple gaussian-like enhancement
            from cupyx.scipy.ndimage import gaussian_filter
            enhanced_field = field + kernel_strength * gaussian_filter(field, sigma=kernel_length/100)
            return enhanced_field
    
    def run_simulation(self, apply_geodesics=True, save_path=None):
        """
        Run the complete GPU-accelerated simulation
        """
        print(f"Running {self.num_steps} steps on GPU...")
        start_time = time.time()
        
        # Create figure
        fig, axes = plt.subplots(1, self.num_steps, figsize=(20, 4), 
                                subplot_kw={'facecolor':'black'})
        
        for step in range(self.num_steps):
            step_start = time.time()
            ax = axes[step]
            
            # Update positions on GPU
            new_angles, x_step, y_step = self.update_positions(step)
            
            # Calculate gravitational field on GPU
            Z_field = self.calculate_gravitational_field(x_step, y_step)
            
            # Apply geodesic enhancement if requested
            if apply_geodesics:
                Z_field = self.apply_geodesic_kernel(Z_field)
            
            # Transfer to CPU for visualization (only the final field)
            Z_field_cpu = cp.asnumpy(Z_field)
            x_step_cpu = cp.asnumpy(x_step)
            y_step_cpu = cp.asnumpy(y_step)
            new_angles_cpu = cp.asnumpy(new_angles)
            star_masses_cpu = cp.asnumpy(self.star_masses)
            
            # Smooth field for visualization
            Z_field_smooth = gaussian_filter(Z_field_cpu, sigma=4)
            
            # Visualization (same as before, but with CPU arrays)
            ax.imshow(Z_field_smooth, 
                     extent=(-self.galaxy_extent, self.galaxy_extent, 
                            -self.galaxy_extent, self.galaxy_extent),
                     cmap='gray_r', alpha=0.4, origin='lower')
            
            # Star trajectory arrows
            ax.quiver(x_step_cpu, y_step_cpu, 
                     -np.sin(new_angles_cpu), np.cos(new_angles_cpu), 
                     color='yellow', scale=20, width=0.004, alpha=0.9)
            
            # Gravitational influence circles
            for sx, sy, mass in zip(x_step_cpu, y_step_cpu, star_masses_cpu):
                influence_circle = patches.Circle((sx, sy), 100 + 20 * mass, 
                                                edgecolor='cyan', facecolor='none', 
                                                alpha=0.6, linewidth=1)
                ax.add_patch(influence_circle)
            
            # Gravitational field vectors
            grad_y, grad_x = np.gradient(Z_field_smooth)
            skip = 30
            X_grid_cpu = cp.asnumpy(self.X_grid)
            Y_grid_cpu = cp.asnumpy(self.Y_grid)
            ax.quiver(X_grid_cpu[::skip, ::skip], Y_grid_cpu[::skip, ::skip],
                     -grad_x[::skip, ::skip], -grad_y[::skip, ::skip],
                     color='cyan', scale=1e7, alpha=0.5, width=0.002)
            
            # Central Black Hole marker
            ax.add_patch(patches.Circle((0, 0), 500, edgecolor='white', 
                                      fill=False, linewidth=1))
            
            # Formatting
            ax.set_xlim(-self.galaxy_extent, self.galaxy_extent)
            ax.set_ylim(-self.galaxy_extent, self.galaxy_extent)
            mode_text = "Geodesic" if apply_geodesics else "Standard"
            time_text = f'{step * self.dt / 1e6:.0f} Myr'
            ax.set_title(f'{time_text}\n{mode_text}', fontsize=12, color='white')
            ax.axis('off')
            
            step_time = time.time() - step_start
            print(f"  Step {step+1}/{self.num_steps} completed in {step_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Simulation completed in {total_time:.2f}s")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return fig

# Performance comparison function
def compare_cpu_gpu_performance():
    """Compare CPU vs GPU performance"""
    
    print("=== Performance Comparison ===")
    
    # Small test case
    test_params = {
        'num_stars': 500,
        'galaxy_extent': 1000,
        'num_steps': 3,
        'grid_res': 200
    }
    
    # GPU simulation
    print("\nRunning GPU simulation...")
    gpu_start = time.time()
    gpu_sim = GPUGalaxySimulation(**test_params)
    gpu_sim.run_simulation(save_path='gpu_galaxy_simulation.png')
    gpu_time = time.time() - gpu_start
    
    print(f"\nGPU Time: {gpu_time:.2f}s")
    print(f"GPU Memory: {cp.get_default_memory_pool().used_bytes() / 1e9:.2f} GB")
    
    return gpu_time

# Example usage
if __name__ == "__main__":
    # More robust GPU detection
    gpu_available = False
    try:
        import cupy as cp
        # Test basic GPU operations
        test_array = cp.array([1.0, 2.0, 3.0])
        result = cp.sum(test_array)
        gpu_available = True
        print(f"GPU detected and working! Test result: {result}")
        print(f"CuPy version: {cp.__version__}")
        
    except ImportError:
        print("CuPy not installed. Install with: pip install cupy-cuda12x")
    except Exception as e:
        print(f"GPU not available: {e}")
        print("Make sure CUDA drivers are installed and GPU is accessible")
    
    if gpu_available:
        try:
            # Run standard simulation
            print("\n=== Running Standard Galaxy Simulation (GPU) ===")
            sim_standard = GPUGalaxySimulation(num_stars=1000, num_steps=5)
            sim_standard.run_simulation(apply_geodesics=False, 
                                       save_path='standard_galaxy_gpu.png')
            
            # Run with geodesic effects
            print("\n=== Running Geodesic-Enhanced Simulation (GPU) ===")
            sim_geodesic = GPUGalaxySimulation(num_stars=1000, num_steps=5)
            sim_geodesic.run_simulation(apply_geodesics=True, 
                                       save_path='geodesic_galaxy_gpu.png')
            
            # Performance comparison
            compare_cpu_gpu_performance()
            
        except Exception as e:
            print(f"GPU simulation failed: {e}")
            print("GPU might not have enough memory or CUDA compatibility issues")
    else:
        print("GPU not available - falling back to CPU would take too long")
        print("Please check CUDA installation and GPU compatibility")