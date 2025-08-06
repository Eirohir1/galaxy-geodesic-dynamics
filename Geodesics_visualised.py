import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math

# === PARAMETERS ===
N = 2000  # Raise this as high as VRAM/memory allows for your GPU
GALAXY_RADIUS = 3e19
CUTOFF = 5e18
CUTOFF_SQ = CUTOFF ** 2
np.random.seed(42)

# === Generate star positions (on CPU) ===
radii = np.random.exponential(GALAXY_RADIUS / 4, N)
radii = np.clip(radii, 1e16, GALAXY_RADIUS)
angles = np.random.uniform(0, 2 * np.pi, N)

positions = np.zeros((N, 2), dtype=np.float32)
positions[:, 0] = radii * np.cos(angles)
positions[:, 1] = radii * np.sin(angles)

# === Estimate upper bound for number of overlap pairs ===
MAX_PAIRS = N * 60  # Most galaxies are sparse, overestimate

# === Allocate device memory ===
d_positions = cuda.to_device(positions)
d_pairs = cuda.device_array((MAX_PAIRS, 2), dtype=np.int32)
d_pair_count = cuda.to_device(np.array([0], dtype=np.int32))

# === CUDA Kernel ===
@cuda.jit
def compute_pairs(pos, cutoff_sq, pairs, pair_count):
    i = cuda.grid(1)
    n = pos.shape[0]
    local_count = 0  # Each thread's local counter
    for j in range(i + 1, n):
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist_sq = dx * dx + dy * dy
        if dist_sq < cutoff_sq:
            idx = cuda.atomic.add(pair_count, 0, 1)
            if idx < pairs.shape[0]:
                pairs[idx, 0] = i
                pairs[idx, 1] = j

# === Launch GPU kernel ===
threads_per_block = 128
blocks = (N + threads_per_block - 1) // threads_per_block
compute_pairs[blocks, threads_per_block](d_positions, CUTOFF_SQ, d_pairs, d_pair_count)

# === Copy results back ===
pair_count = d_pair_count.copy_to_host()[0]
pairs = d_pairs.copy_to_host()[:pair_count]
positions_kpc = positions / 3.086e16  # Convert to kiloparsecs

# === Plot ===
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('black')

for i, j in pairs:
    ax.plot([positions_kpc[i, 0], positions_kpc[j, 0]],
            [positions_kpc[i, 1], positions_kpc[j, 1]],
            color='cyan', alpha=0.11, linewidth=0.7, zorder=1)

print(f"✅ Total geodesic overlap lines drawn: {pair_count}")

# Draw stars
ax.scatter(positions_kpc[:, 0], positions_kpc[:, 1],
           color='magenta', s=30, alpha=0.92, label='Stars', zorder=2)

# Draw CBH
ax.scatter(0, 0, color='yellow', marker='*', s=180, label='CBH', zorder=3)

ax.set_xlabel('X (kpc)')
ax.set_ylabel('Y (kpc)')
ax.set_title(f'Geodesic Overlap Web (CUDA Accelerated)\nN={N}, cutoff ~160 kpc')
ax.legend(facecolor='black', framealpha=1, loc='lower left', fontsize=10)
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
