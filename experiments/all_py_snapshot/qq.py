# Re-importing libraries after reset
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches

# Reinitialize parameters (due to environment reset)
num_stars = 1000
galaxy_extent = 2000  # parsecs
cbh_mass = 1e10
G = 4.30091e-6
num_steps = 5
dt = 5e6

# Initial star positions and velocities
np.random.seed(42)
star_radii = np.sqrt(np.random.uniform(1000**2, galaxy_extent**2, num_stars))
star_angles = np.random.uniform(0, 2 * np.pi, num_stars)
star_speeds = np.sqrt(G * cbh_mass / (star_radii / 1000))

star_masses = np.random.uniform(0.5, 5.0, num_stars)

# Gravitational field grid setup
grid_res = 500
X_grid, Y_grid = np.meshgrid(np.linspace(-galaxy_extent, galaxy_extent, grid_res),
                             np.linspace(-galaxy_extent, galaxy_extent, grid_res))

# Visualization with gravitational influence circles
fig, axes = plt.subplots(1, num_steps, figsize=(20, 4), subplot_kw={'facecolor':'black'})

for step in range(num_steps):
    ax = axes[step]
    
    # Update positions (simple circular motion assumption)
    angle_change = (star_speeds / star_radii) * (dt * step * 3.154e+7 / (3.086e+13))
    new_angles = star_angles + angle_change
    x_step = star_radii * np.cos(new_angles)
    y_step = star_radii * np.sin(new_angles)

    # Global gravitational field (CBH + stars)
    Z_global = -cbh_mass / np.sqrt(X_grid**2 + Y_grid**2 + 1e3)
    for sx, sy in zip(x_step, y_step):
        Z_global -= 1e5 / np.sqrt((X_grid - sx)**2 + (Y_grid - sy)**2 + 1e3)
    Z_global_smooth = gaussian_filter(Z_global, sigma=4)

    # Overlay curvature gradient
    ax.imshow(Z_global_smooth, extent=(-galaxy_extent, galaxy_extent, -galaxy_extent, galaxy_extent),
              cmap='gray_r', alpha=0.4, origin='lower')

    # Star trajectory arrows
    ax.quiver(x_step, y_step, -np.sin(new_angles), np.cos(new_angles), color='yellow', scale=20, width=0.004, alpha=0.9)

    # Gravitational influence circles (scaled by star mass)
    for sx, sy, mass in zip(x_step, y_step, star_masses):
        influence_circle = patches.Circle((sx, sy), 100 + 20 * mass, edgecolor='cyan', facecolor='none', alpha=0.6, linewidth=1)
        ax.add_patch(influence_circle)

    # Gravitational current vectors
    grad_y, grad_x = np.gradient(Z_global_smooth)
    skip = 30
    ax.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip],
              -grad_x[::skip, ::skip], -grad_y[::skip, ::skip],
              color='cyan', scale=1e7, alpha=0.5, width=0.002)

    # Central Black Hole marker
    ax.add_patch(patches.Circle((0, 0), 500, edgecolor='white', fill=False, linewidth=1))

    # Formatting
    ax.set_xlim(-galaxy_extent, galaxy_extent)
    ax.set_ylim(-galaxy_extent, galaxy_extent)
    ax.set_title(f'{step * dt / 1e6:.0f} Myr', fontsize=12, color='white')
    ax.axis('off')

plt.tight_layout()
output_final_path = 'C:\\Users\\vinny\\Documents\\Starsim\\outputs\\qq.png'
plt.savefig(output_final_path, dpi=300, bbox_inches='tight')
plt.show()
