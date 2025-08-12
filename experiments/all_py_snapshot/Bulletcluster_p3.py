import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Output folder setup ===
output_dir = "Geodesic_BulletCluster_Sims/Phase3_Results"
os.makedirs(output_dir, exist_ok=True)

# === Constants ===
G = 4.302e-6  # gravitational constant in kpc⋅(km/s)^2⋅M☉^−1

# === Cluster Properties ===
M1 = M2 = 1e15  # Solar masses
pos1 = np.array([-350, 0])
pos2 = np.array([350, 0])

# === Grid Setup ===
grid_size = 200
extent = 1000
x = np.linspace(-extent, extent, grid_size)
y = np.linspace(-extent, extent, grid_size)
X, Y = np.meshgrid(x, y)

# === Gravitational Potential ===
def potential_field(X, Y):
    r1 = np.sqrt((X - pos1[0])**2 + (Y - pos1[1])**2)
    r2 = np.sqrt((X - pos2[0])**2 + (Y - pos2[1])**2)
    return -G * (M1 / (r1 + 1e-3) + M2 / (r2 + 1e-3))

phi = potential_field(X, Y)

# === Gradient Field ===
grad_y, grad_x = np.gradient(phi, extent * 2 / grid_size)
gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

# === Save Gradient CSV ===
grad_df = pd.DataFrame({
    'x': X.flatten(),
    'y': Y.flatten(),
    'grad_x': grad_x.flatten(),
    'grad_y': grad_y.flatten(),
    'magnitude': gradient_magnitude.flatten()
})
grad_df.to_csv(os.path.join(output_dir, "bullet_cluster_phase3_gradient_field.csv"), index=False)

# === Simulate Matter ===
def gaussian_cloud(center, sigma, N):
    return np.random.normal(loc=center, scale=sigma, size=(N, 2))

stars1 = gaussian_cloud(pos1, sigma=150, N=1000)
stars2 = gaussian_cloud(pos2, sigma=150, N=1000)
stars = np.vstack([stars1, stars2])

gas = gaussian_cloud(center=[0, 0], sigma=300, N=3000)

# === Save Star + Gas CSVs ===
pd.DataFrame(stars, columns=['x', 'y']).to_csv(os.path.join(output_dir, "bullet_cluster_phase3_stars.csv"), index=False)
pd.DataFrame(gas, columns=['x', 'y']).to_csv(os.path.join(output_dir, "bullet_cluster_phase3_gas.csv"), index=False)

# === Plot 1: Curvature Potential Map ===
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, phi, levels=50, cmap='inferno')
plt.colorbar(label="Potential (km²/s²)")
plt.title("Gravitational Curvature Map (Potential Field) - Phase 3")
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase3_curvature_map.png"))
plt.close()

# === Plot 2: Gradient Vectors ===
plt.figure(figsize=(8, 6))
plt.quiver(X, Y, -grad_x, -grad_y, color='cyan', scale=2e-7, width=0.002)
plt.title("Curvature Gradient Field (Lensing Direction Proxy) - Phase 3")
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase3_gradient_vectors.png"))
plt.close()

# === Plot 3: Star Distribution ===
plt.figure(figsize=(8, 6))
plt.scatter(stars[:, 0], stars[:, 1], s=5, color='yellow', label='Star Clusters')
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.title("Simulated Star Cluster Distribution - Phase 3")
plt.legend()
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase3_star_distribution.png"))
plt.close()

# === Plot 4: Gas Distribution ===
plt.figure(figsize=(8, 6))
plt.scatter(gas[:, 0], gas[:, 1], s=2, color='blue', label='Gas Cloud')
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.title("Simulated Gas Distribution - Phase 3")
plt.legend()
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase3_gas_distribution.png"))
plt.close()

# === Plot 5: Combined Matter Map ===
plt.figure(figsize=(10, 8))
plt.scatter(gas[:, 0], gas[:, 1], s=1, color='blue', alpha=0.4, label='Gas Cloud')
plt.scatter(stars[:, 0], stars[:, 1], s=3, color='yellow', label='Star Clusters')
plt.xlabel("kpc")
plt.ylabel("kpc")
plt.title("Simulated Matter Distribution - Phase 3")
plt.legend()
plt.savefig(os.path.join(output_dir, "bullet_cluster_phase3_matter_distribution.png"))
plt.close()

print(f"✅ Phase 3 simulation complete.\nAll files saved to: {output_dir}")
