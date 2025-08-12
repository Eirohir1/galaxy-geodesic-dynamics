import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter, sobel

# === CONFIGURATION ===
GRID_SIZE = 512
NUM_PARTICLES = 5000
NUM_STEPS = 200
STEP_SIZE = 0.8
CURVATURE_FILE = "curvature_map.npy"
OUTPUT_DIR = "universe_evolution_frames"
SEED_FILE = "initial_seed_positions.npy"

# === 1. LOAD OR GENERATE CURVATURE FIELD ===
if os.path.exists(CURVATURE_FILE):
    curvature = np.load(CURVATURE_FILE)
    print("[INFO] Loaded curvature from file.")
else:
    print("[ERROR] Curvature map not found.")
    exit(1)

# Optional: Smooth the curvature to prevent noise amplification
curvature = gaussian_filter(curvature, sigma=1.2)

# === 2. COMPUTE GRADIENT FIELD FROM CURVATURE ===
grad_y = sobel(curvature, axis=0, mode='reflect')
grad_x = sobel(curvature, axis=1, mode='reflect')

magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-8  # avoid div by zero
grad_x /= magnitude
grad_y /= magnitude

# === 3. SEED PARTICLES IN FIELD ===
if os.path.exists(SEED_FILE):
    particles = np.load(SEED_FILE)
    print(f"[INFO] Loaded {len(particles)} seed particles from file.")
else:
    # Uniform random seeding if none provided
    particles = np.random.rand(NUM_PARTICLES, 2) * GRID_SIZE
    np.save(SEED_FILE, particles)
    print(f"[INFO] Saved new seed positions to {SEED_FILE}.")

# === 4. SETUP OUTPUT ===
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 5. RUN TIME EVOLUTION ===
for step in range(NUM_STEPS):
    # Interpolate gradient at each particle’s location
    px = np.clip(particles[:, 0].astype(int), 0, GRID_SIZE - 1)
    py = np.clip(particles[:, 1].astype(int), 0, GRID_SIZE - 1)

    dx = grad_x[py, px]
    dy = grad_y[py, px]

    # Update particle positions (move along gradient)
    particles[:, 0] -= dx * STEP_SIZE
    particles[:, 1] -= dy * STEP_SIZE

    # Clamp within bounds
    particles = np.clip(particles, 0, GRID_SIZE - 1)

    # Plot and save frame
    plt.figure(figsize=(6, 6))
    plt.imshow(curvature, cmap='plasma', alpha=0.5)
    plt.scatter(particles[:, 0], particles[:, 1], s=1, color='white', alpha=0.4)
    plt.title(f"Geodesic Universe Evolution - Step {step}")
    plt.axis('off')

    frame_path = os.path.join(OUTPUT_DIR, f"step_{step:03}.png")
    plt.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close()

    if step % 10 == 0:
        print(f"[STEP {step}] Saved: {frame_path}")

# === 6. SAVE FINAL PARTICLE STATE ===
np.save(os.path.join(OUTPUT_DIR, "final_particle_positions.npy"), particles)
print("[DONE] Universe evolution complete. Final state saved.")
