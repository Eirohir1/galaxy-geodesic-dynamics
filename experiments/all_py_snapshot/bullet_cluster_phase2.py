import numpy as np
import matplotlib.pyplot as plt
import os

# === SETUP ===
output_dir = "Geodesic_BulletCluster_Sims/phase2_post_collision"
os.makedirs(output_dir, exist_ok=True)

# === CLUSTER DEFINITIONS (post-collision) ===
G = 4.302e-6  # kpc * (km/s)^2 / Msun

mass1 = 5.5e14  # Msun
mass2 = 5.5e14  # Msun

pos1 = np.array([-200, 0])  # kpc
pos2 = np.array([200, 0])   # kpc

# === GRID FOR FIELD CALCULATION ===
grid_size = 1000  # kpc
resolution = 500
x = np.linspace(-grid_size, grid_size, resolution)
y = np.linspace(-grid_size, grid_size, resolution)
X, Y = np.meshgrid(x, y)

# === CALCULATE GRAVITATIONAL POTENTIAL ===
def potential_field(mass, pos, X, Y):
    dx = X - pos[0]
    dy = Y - pos[1]
    r = np.sqrt(dx**2 + dy**2) + 1e-3
    return -G * mass / r

phi1 = potential_field(mass1, pos1, X, Y)
phi2 = potential_field(mass2, pos2, X, Y)
phi_total = phi1 + phi2

# === SAVE CURVATURE MAP ===
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, phi_total, levels=100, cmap='inferno')
plt.colorbar(label='Potential (km²/s²)')
plt.title('Gravitational Curvature Map (Potential Field) - Phase 2')
plt.xlabel('kpc')
plt.ylabel('kpc')
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase2_curvature_map.png"))
plt.close()

# === CALCULATE GRADIENT FIELD (lensing proxy) ===
gx, gy = np.gradient(-phi_total)
plt.figure(figsize=(8, 6))
plt.quiver(X[::10, ::10], Y[::10, ::10], gx[::10, ::10], gy[::10, ::10], color='cyan')
plt.title("Curvature Gradient Field (Lensing Direction Proxy) - Phase 2")
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase2_gradient_vectors.png"))
plt.close()

# === SAVE GRADIENT FIELD TO CSV ===
np.savetxt(os.path.join(output_dir, "bullet_cluster_phase2_gradient_field.csv"),
           np.dstack((gx, gy)).reshape(-1, 2), delimiter=',', header="gx,gy", comments='')

# === SIMULATE STAR CLUSTERS ===
n_stars = 500
star1 = np.random.normal(loc=pos1, scale=80, size=(n_stars, 2))
star2 = np.random.normal(loc=pos2, scale=80, size=(n_stars, 2))
stars = np.vstack((star1, star2))

plt.figure(figsize=(8, 6))
plt.scatter(stars[:, 0], stars[:, 1], s=1, color='yellow', label="Star Clusters")
plt.title("Simulated Star Cluster Distribution - Phase 2")
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.legend()
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase2_star_distribution.png"))
plt.close()

np.savetxt(os.path.join(output_dir, "bullet_cluster_phase2_stars.csv"),
           stars, delimiter=',', header="x,y", comments='')

# === SIMULATE DIFFUSE GAS CLOUD ===
n_gas = 2000
gas1 = np.random.normal(loc=pos1 + np.array([100, 0]), scale=300, size=(n_gas, 2))
gas2 = np.random.normal(loc=pos2 - np.array([100, 0]), scale=300, size=(n_gas, 2))
gas = np.vstack((gas1, gas2))

plt.figure(figsize=(8, 6))
plt.scatter(gas[:, 0], gas[:, 1], s=1, color='blue', alpha=0.5, label="Gas Cloud")
plt.title("Simulated Gas Distribution - Phase 2")
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.legend()
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase2_gas_distribution.png"))
plt.close()

np.savetxt(os.path.join(output_dir, "bullet_cluster_phase2_gas.csv"),
           gas, delimiter=',', header="x,y", comments='')

# === COMBINED MATTER MAP ===
plt.figure(figsize=(10, 8))
plt.scatter(gas[:, 0], gas[:, 1], s=1, color='blue', alpha=0.3, label="Gas Cloud")
plt.scatter(stars[:, 0], stars[:, 1], s=2, color='yellow', label="Star Clusters")
plt.title("Simulated Matter Distribution - Phase 2")
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.legend()
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase2_matter_distribution.png"))
plt.close()
