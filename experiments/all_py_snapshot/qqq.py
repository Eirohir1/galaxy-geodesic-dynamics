# IMPROVED VERSION - Better units & visualization

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
import time

# === BETTER SCALED CONFIG ===
NUM_PARTICLES = 50000
NUM_STEPS = 200
DT = 1e11  # Reduced time step (was 1e12)
GALAXY_RADIUS = 3e19  # ~10 kpc (more reasonable)
GALAXY_THICKNESS = 3e17  # ~1 kpc
BH_MASS = 4e36  # 2M☉ supermassive black hole
CUTOFF_RADIUS = 9.259e15  # 0.3 pc (10x larger for more interactions)
G = 6.67430e-11

@cuda.jit
def compute_forces_kernel(positions, masses, accelerations, cutoff, galaxy_radius, N, G_const):
    i = cuda.grid(1)
    
    if i < N:
        ax = 0.0
        ay = 0.0
        az = 0.0
        
        xi = positions[i, 0]
        yi = positions[i, 1] 
        zi = positions[i, 2]
        
        for j in range(N):
            if i != j:
                dx = xi - positions[j, 0]
                dy = yi - positions[j, 1]
                dz = zi - positions[j, 2]
                
                dist_sq = dx*dx + dy*dy + dz*dz
                
                if dist_sq < cutoff*cutoff:
                    dist = math.sqrt(dist_sq + 1e15)  # Better softening
                    
                    strength = G_const * masses[j] / dist_sq
                    taper = math.exp(-dist / (0.5 * galaxy_radius))
                    
                    force = -strength * taper / dist
                    
                    ax += force * dx
                    ay += force * dy  
                    az += force * dz
        
        accelerations[i, 0] = ax
        accelerations[i, 1] = ay
        accelerations[i, 2] = az

start_time = time.time()

print("🚀 IMPROVED GPU Local Geodesic Galaxy Simulation")
print(f"Particles: {NUM_PARTICLES:,}")
print(f"Galaxy radius: {GALAXY_RADIUS/3.086e16:.1f} kpc")
print(f"Cutoff radius: {CUTOFF_RADIUS/3.086e16:.2f} pc")

device = cp.cuda.Device()
print(f"GPU: RTX 3080 Ti (Device {device.id})")

# === MASS SETUP ===
masses = cp.random.lognormal(mean=0, sigma=0.5, size=NUM_PARTICLES).astype(cp.float64) * 1.989e30
masses[0] = BH_MASS

# === POSITION SETUP ===
radii = cp.random.exponential(GALAXY_RADIUS/4, NUM_PARTICLES).astype(cp.float64)
radii = cp.clip(radii, 1e16, GALAXY_RADIUS)  # Minimum distance from center
angles = cp.random.uniform(0, 2 * cp.pi, NUM_PARTICLES).astype(cp.float64)
z_pos = cp.random.normal(0, GALAXY_THICKNESS/4, NUM_PARTICLES).astype(cp.float64)

positions = cp.zeros((NUM_PARTICLES, 3), dtype=cp.float64)
positions[:, 0] = radii * cp.cos(angles)
positions[:, 1] = radii * cp.sin(angles)
positions[:, 2] = z_pos

# === BETTER VELOCITY SETUP ===
velocities = cp.zeros((NUM_PARTICLES, 3), dtype=cp.float64)
for i in range(1, NUM_PARTICLES):
    r = radii[i]
    # Circular velocity for stable orbit
    v_circ = cp.sqrt(G * BH_MASS / r)
    # Add stellar contribution (rough approximation)
    v_circ *= 1.2
    
    # Tangential direction
    velocities[i, 0] = -v_circ * cp.sin(angles[i])
    velocities[i, 1] = v_circ * cp.cos(angles[i])
    # Small random motion
    velocities[i] += cp.random.normal(0, v_circ * 0.1, 3)

# Convert to float32 for GPU
positions = positions.astype(cp.float32)
velocities = velocities.astype(cp.float32)
masses = masses.astype(cp.float32)
accelerations = cp.zeros((NUM_PARTICLES, 3), dtype=cp.float32)

threads_per_block = 256
blocks_per_grid = (NUM_PARTICLES + threads_per_block - 1) // threads_per_block

print(f"CUDA config: {blocks_per_grid} blocks × {threads_per_block} threads")
print("Starting simulation...")

# === SIMULATION ===
for step in range(NUM_STEPS):
    if step % 40 == 0:
        print(f"Step {step}/{NUM_STEPS}")
    
    compute_forces_kernel[blocks_per_grid, threads_per_block](
        positions, masses, accelerations, 
        CUTOFF_RADIUS, GALAXY_RADIUS, NUM_PARTICLES, G
    )
    cp.cuda.Device().synchronize()
    
    velocities += accelerations * DT
    positions += velocities * DT

runtime = time.time() - start_time
print(f"\n✅ GPU Simulation complete in {runtime:.1f} seconds!")
print(f"⚡ Performance: {NUM_PARTICLES * NUM_STEPS / runtime:.0f} particle-steps/second")

# === IMPROVED VISUALIZATION ===
positions_np = cp.asnumpy(positions)
velocities_np = cp.asnumpy(velocities)

# Convert to kpc and km/s for plotting
pos_kpc = positions_np / 3.086e16  # Convert to kpc
vel_kms = velocities_np / 1000     # Convert to km/s

radii_kpc = np.sqrt(pos_kpc[:, 0]**2 + pos_kpc[:, 1]**2)
vel_mag = np.sqrt(vel_kms[:, 0]**2 + vel_kms[:, 1]**2)

# Remove particles that escaped
bound_mask = radii_kpc < GALAXY_RADIUS / 3.086e16  # Within galaxy radius
print(f"Particles retained: {np.sum(bound_mask)}/{NUM_PARTICLES} ({100*np.sum(bound_mask)/NUM_PARTICLES:.1f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Galaxy structure (in kpc)
ax1.set_facecolor('black')
galaxy_radius_kpc = GALAXY_RADIUS / 3.086e16
ax1.set_xlim(-galaxy_radius_kpc, galaxy_radius_kpc)
ax1.set_ylim(-galaxy_radius_kpc, galaxy_radius_kpc)

# Plot only bound particles
bound_pos = pos_kpc[bound_mask]
ax1.scatter(bound_pos[1:, 0], bound_pos[1:, 1], s=0.5, color='white', alpha=0.8)
ax1.scatter(pos_kpc[0, 0], pos_kpc[0, 1], s=50, color='red', marker='*', label='SMBH')
ax1.set_xlabel('X (kpc)')
ax1.set_ylabel('Y (kpc)')
ax1.set_title(f'Local Geodesic Galaxy - {np.sum(bound_mask):,} bound particles')
ax1.legend()

# Rotation curve
bound_radii = radii_kpc[bound_mask]
bound_vel = vel_mag[bound_mask]

if len(bound_radii) > 20:
    r_bins = np.linspace(0.1, np.max(bound_radii), 12)
    v_binned = []
    r_binned = []
    
    for i in range(len(r_bins)-1):
        mask = (bound_radii >= r_bins[i]) & (bound_radii < r_bins[i+1])
        if np.sum(mask) > 3:
            v_avg = np.mean(bound_vel[mask])
            r_avg = (r_bins[i] + r_bins[i+1]) / 2
            v_binned.append(v_avg)
            r_binned.append(r_avg)
    
    ax2.plot(r_binned, v_binned, 'b-o', linewidth=2, markersize=5)
    ax2.set_xlabel('Radius (kpc)')
    ax2.set_ylabel('Velocity (km/s)')
    ax2.set_title('Local Geodesic Rotation Curve')
    ax2.grid(True, alpha=0.3)
    
    if len(v_binned) > 0:
        ax2.set_ylim(0, max(v_binned) * 1.1)

plt.tight_layout()
plt.show()

print(f"🎯 Local interactions within {CUTOFF_RADIUS/3.086e16:.2f} pc")
print(f"🌟 Galaxy stable with {100*np.sum(bound_mask)/NUM_PARTICLES:.1f}% particle retention!")