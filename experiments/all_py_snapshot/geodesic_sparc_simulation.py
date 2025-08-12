import os
import cupy as cp
import numpy as np
import time
import math

# === CONFIG ===
NUM_PARTICLES = 50000  # Number of particles per galaxy
NUM_STEPS = 200  # Simulation steps
DT = 1e11  # Time step
G = 6.67430e-11  # Gravitational constant
SUCCESS_THRESHOLD = 0.1  # 10% velocity difference allowed for a "successful" fit

# Directory path to the .dat files
DAT_FILES_PATH = r'C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts'

# Summary report file
SUMMARY_REPORT_PATH = r'C:\Users\vinny\Documents\geodesic_theory_package\Python Scripts\simulation_summary_report.txt'

# Function to load data from .dat files
def load_rotmod_data(file_path):
    print(f"Loading data from {file_path}")
    data = np.loadtxt(file_path, comments="#")  # Skip comments (headers)
    r_kpc = data[:, 0]  # Radius in kpc
    v_obs = data[:, 1]  # Observed velocities in km/s
    dv_obs = data[:, 2]  # Velocity errors in km/s
    return r_kpc, v_obs, dv_obs

# === GEODESIC SIMULATION KERNEL ===
@cuda.jit
def compute_geodesic_kernel(positions, velocities, accelerations, cutoff, galaxy_radius, N, G_const):
    i = cuda.grid(1)
    
    if i < N:
        ax = 0.0
        ay = 0.0
        az = 0.0
        
        xi = positions[i, 0]
        yi = positions[i, 1]
        zi = positions[i, 2]
        
        # Geodesic motion: update acceleration based on gravitational potential
        for j in range(N):
            if i != j:
                dx = xi - positions[j, 0]
                dy = yi - positions[j, 1]
                dz = zi - positions[j, 2]
                
                dist_sq = dx*dx + dy*dy + dz*dz
                if dist_sq < cutoff * cutoff:
                    dist = math.sqrt(dist_sq + 1e15)  # Softening
                    
                    strength = -G_const * galaxy_radius / dist_sq  # Use galaxy mass for simplicity
                    taper = math.exp(-dist / (0.5 * galaxy_radius))
                    
                    force = strength * taper / dist
                    ax += force * dx
                    ay += force * dy
                    az += force * dz
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az

# === MAIN SIMULATION FUNCTION ===
def run_simulation_for_galaxies(dat_files):
    success_count = 0  # Counter for successful fits
    total_galaxies = 0  # Counter for total galaxies processed

    # Open the summary report file for writing
    with open(SUMMARY_REPORT_PATH, 'w') as summary_file:
        summary_file.write("Geodesic Galaxy Simulation Summary Report\n")
        summary_file.write("=" * 50 + "\n")

        for file_name in os.listdir(dat_files):
            if file_name.endswith(".dat"):
                total_galaxies += 1
                file_path = os.path.join(dat_files, file_name)
                print(f"\n🚀 Running simulation for galaxy data: {file_name}")

                # Load the galaxy data
                r_kpc, v_obs, dv_obs = load_rotmod_data(file_path)

                # Initialize galaxy parameters
                galaxy_radius = np.max(r_kpc) * 3.086e19  # kpc to meters
                galaxy_mass = 1e11  # Placeholder mass (adjust as needed)

                # Initialize particles
                positions, velocities, masses = initialize_particles(NUM_PARTICLES, galaxy_radius, galaxy_mass)

                # Prepare GPU arrays
                positions = positions.astype(cp.float32)
                velocities = velocities.astype(cp.float32)
                masses = masses.astype(cp.float32)
                accelerations = cp.zeros((NUM_PARTICLES, 3), dtype=cp.float32)

                threads_per_block = 256
                blocks_per_grid = (NUM_PARTICLES + threads_per_block - 1) // threads_per_block
                
                # Run the simulation
                for step in range(NUM_STEPS):
                    if step % 40 == 0:
                        print(f"Step {step}/{NUM_STEPS} for {file_name}")
                    
                    # Run geodesic kernel
                    compute_geodesic_kernel[blocks_per_grid, threads_per_block](
                        positions, velocities, accelerations, 
                        9.259e15, galaxy_radius, NUM_PARTICLES, G
                    )
                    cp.cuda.Device().synchronize()
                    
                    # Update positions and velocities
                    velocities += accelerations * DT
                    positions += velocities * DT

                # Analyze the results for this galaxy and write to summary report
                is_successful, summary = analyze_results(file_name, r_kpc, v_obs, velocities, positions)
                if is_successful:
                    success_count += 1
                summary_file.write(summary + "\n")

        # Final Success Rate
        success_rate = (success_count / total_galaxies) * 100
        summary_file.write("\n" + "=" * 50 + "\n")
        summary_file.write(f"Total Galaxies: {total_galaxies}\n")
        summary_file.write(f"Successful Fits: {success_count}\n")
        summary_file.write(f"Success Rate: {success_rate:.2f}%\n")

def initialize_particles(num_particles, galaxy_radius, galaxy_mass):
    # Random particle initialization within galaxy radius
    radii = cp.random.exponential(galaxy_radius / 4, num_particles).astype(cp.float64)
    radii = cp.clip(radii, 1e16, galaxy_radius)  # Minimum distance from center
    angles = cp.random.uniform(0, 2 * cp.pi, num_particles).astype(cp.float64)
    z_pos = cp.random.normal(0, galaxy_radius / 4, num_particles).astype(cp.float64)

    positions = cp.zeros((num_particles, 3), dtype=cp.float64)
    positions[:, 0] = radii * cp.cos(angles)
    positions[:, 1] = radii * cp.sin(angles)
    positions[:, 2] = z_pos

    # Circular velocity for stable orbit based on geodesic motion (simplified)
    velocities = cp.zeros((num_particles, 3), dtype=cp.float64)
    for i in range(1, num_particles):
        r = radii[i]
        v_circ = cp.sqrt(G * galaxy_mass / r)
        velocities[i, 0] = -v_circ * cp.sin(angles[i])
        velocities[i, 1] = v_circ * cp.cos(angles[i])
        velocities[i] += cp.random.normal(0, v_circ * 0.1, 3)
    
    masses = cp.ones(num_particles) * galaxy_mass / num_particles  # Equal mass particles for simplicity
    
    return positions, velocities, masses

def analyze_results(galaxy_name, r_kpc, v_obs, velocities, positions):
    # Convert positions and velocities to NumPy arrays
    pos_np = cp.asnumpy(positions)
    vel_np = cp.asnumpy(velocities)

    # Convert radii and velocities to kpc and km/s
    radii_kpc = np.sqrt(pos_np[:, 0]**2 + pos_np[:, 1]**2) / 3.086e16  # Convert to kpc
    vel_kms = np.sqrt(vel_np[:, 0]**2 + vel_np[:, 1]**2) / 1000  # Convert to km/s

    # Bin the velocities based on the radii (radii_kpc)
    r_bins = np.linspace(np.min(radii_kpc), np.max(radii_kpc), len(v_obs))  # Create bins for comparison
    v_binned = []
    
    for i in range(len(r_bins) - 1):
        # Get the mask for particles within the current bin
        mask = (radii_kpc >= r_bins[i]) & (radii_kpc < r_bins[i + 1])
        if np.sum(mask) > 0:
            # Compute the average velocity for the particles in the current bin
            v_avg = np.mean(vel_kms[mask])
            v_binned.append(v_avg)
    
    # Check if the binned velocities match the observed velocities within the defined threshold
    velocity_diff = np.abs(np.array(v_binned) - v_obs) / v_obs
    is_successful = np.all(velocity_diff < SUCCESS_THRESHOLD)

    # Output the success or failure for this galaxy
    max_radius = np.max(radii_kpc)
    max_velocity = np.max(vel_kms)
    
    summary = f"Galaxy: {galaxy_name}\n"
    summary += f"Max Radius: {max_radius:.2f} kpc\n"
    summary += f"Max Velocity: {max_velocity:.2f} km/s\n"
    summary += f"Particle Retention: {100 * (np.sum(radii_kpc < max_radius) / len(radii_kpc)):.2f}%\n"
    summary += f"Success: {'Yes' if is_successful else 'No'}\n"
    summary += "-" * 50
    
    return is_successful, summary

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Run the simulation for galaxies in your dataset folder
    run_simulation_for_galaxies(DAT_FILES_PATH)
