import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
G = 4.302e-6  # Gravitational constant in (kpc / solar mass) * (km/s)^2

# Cluster parameters
mass1 = 1e15  # Mass of first cluster in solar masses
mass2 = 1e15  # Mass of second cluster
distance = 750  # Separation in kpc
grid_size = 500
extent = 2000  # Total extent of simulation box in kpc
resolution = extent / grid_size

# Grid setup
x = np.linspace(-extent / 2, extent / 2, grid_size)
y = np.linspace(-extent / 2, extent / 2, grid_size)
X, Y = np.meshgrid(x, y)

# Positions of two clusters
cluster1_pos = (-distance / 2, 0)
cluster2_pos = (distance / 2, 0)

# Geodesic potential field (using Newtonian potential for curvature proxy)
def potential_field(xc, yc, M):
    R = np.sqrt((X - xc)**2 + (Y - yc)**2) + 1e-3  # avoid singularity
    return -G * M / R

phi1 = potential_field(*cluster1_pos, mass1)
phi2 = potential_field(*cluster2_pos, mass2)
phi_total = phi1 + phi2

# Gradient of the field (to approximate lensing direction)
grad_y, grad_x = np.gradient(phi_total, resolution, resolution)

# Plot potential map
plt.figure(figsize=(8, 6))
plt.title("Gravitational Curvature Map (Potential Field)")
plt.contourf(X, Y, phi_total, levels=100, cmap='inferno')
plt.colorbar(label='Potential (km²/s²)')
plt.xlabel('kpc')
plt.ylabel('kpc')
plt.savefig("bullet_cluster_phase1_curvature_map.png", dpi=300)
plt.close()

# Plot gradient field (approximating lensing direction)
plt.figure(figsize=(8, 6))
plt.title("Curvature Gradient Field (Lensing Direction Proxy)")
plt.quiver(X[::10, ::10], Y[::10, ::10], -grad_x[::10, ::10], -grad_y[::10, ::10], color='cyan', scale=2e-3)
plt.contour(X, Y, phi_total, levels=30, colors='white', linewidths=0.5)
plt.xlabel('kpc')
plt.ylabel('kpc')
plt.savefig("bullet_cluster_phase1_gradient_vectors.png", dpi=300)
plt.close()

# Simulate baryonic gas cloud (diffuse component)
np.random.seed(42)
n_gas = 1000
gas_x = np.random.normal(loc=0, scale=extent/3, size=n_gas)
gas_y = np.random.normal(loc=0, scale=extent/3, size=n_gas)
gas_density = np.exp(-(gas_x**2 + gas_y**2) / (2 * (extent/4)**2))

# Simulate star particles (bound clusters)
n_stars = 500
stars_x = np.concatenate([
    np.random.normal(loc=cluster1_pos[0], scale=100, size=n_stars//2),
    np.random.normal(loc=cluster2_pos[0], scale=100, size=n_stars//2)
])
stars_y = np.concatenate([
    np.random.normal(loc=cluster1_pos[1], scale=100, size=n_stars//2),
    np.random.normal(loc=cluster2_pos[1], scale=100, size=n_stars//2)
])

# Visualize matter
plt.figure(figsize=(8, 6))
plt.title("Simulated Matter Distribution")
plt.scatter(gas_x, gas_y, c='blue', s=2, alpha=0.3, label='Gas Cloud')
plt.scatter(stars_x, stars_y, c='yellow', s=5, alpha=0.8, label='Star Clusters')
plt.xlabel('kpc')
plt.ylabel('kpc')
plt.legend()
plt.savefig("bullet_cluster_phase1_matter_distribution.png", dpi=300)
plt.close()

# Save data to CSV for review
df_gas = pd.DataFrame({'x': gas_x, 'y': gas_y, 'density': gas_density})
df_stars = pd.DataFrame({'x': stars_x, 'y': stars_y})
df_grad = pd.DataFrame({
    'x': X.flatten(),
    'y': Y.flatten(),
    'grad_x': grad_x.flatten(),
    'grad_y': grad_y.flatten()
})

df_gas.to_csv("bullet_cluster_phase1_gas.csv", index=False)
df_stars.to_csv("bullet_cluster_phase1_stars.csv", index=False)
df_grad.to_csv("bullet_cluster_phase1_gradient_field.csv", index=False)

print("✅ Bullet Cluster Phase 1 simulation complete.")
print("📁 Output files saved:")
print("- bullet_cluster_phase1_curvature_map.png")
print("- bullet_cluster_phase1_gradient_vectors.png")
print("- bullet_cluster_phase1_matter_distribution.png")
print("- bullet_cluster_phase1_gas.csv")
print("- bullet_cluster_phase1_stars.csv")
print("- bullet_cluster_phase1_gradient_field.csv")
