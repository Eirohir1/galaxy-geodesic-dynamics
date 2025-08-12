import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import math
import time # Import time module to measure execution speed

# === PARAMETERS ===
N = 5000  # Significantly Reduced: Fewer stars = faster GPU kernel and much faster plotting
GALAXY_RADIUS = 3e19
CUTOFF = 5e18
CUTOFF_SQ = CUTOFF ** 2
np.random.seed(42)

# Set to True to display the plot window, False to just save and exit immediately
SKIP_PLOTTING = False 
SAVE_PLOT_TO_FILE = True # Set to True to save the plot image even if not displayed

# === Generate star positions (on CPU) ===
radii = np.random.exponential(GALAXY_RADIUS / 4, N)
radii = np.clip(radii, 1e16, GALAXY_RADIUS)
angles = np.random.uniform(0, 2 * np.pi, N)

positions = np.zeros((N, 2), dtype=np.float32)
positions[:, 0] = radii * np.cos(angles)
positions[:, 1] = radii * np.sin(angles)

# === Estimate upper bound for number of overlap pairs ===
MAX_PAIRS = N * 60  # Overestimate for safety

# === Allocate device memory ===
d_positions = cuda.to_device(positions)
d_pairs = cuda.device_array((MAX_PAIRS, 2), dtype=np.int32)
d_pair_count = cuda.to_device(np.array([0], dtype=np.int32))

# === Confirm GPU is visible and report it ===
device = cuda.get_current_device()
print(f"✅ CUDA Device in use: {device.name}, {device.MAX_THREADS_PER_BLOCK} threads/block")

# === CUDA Kernel ===
@cuda.jit
def compute_pairs(pos, cutoff_sq, pairs, pair_count):
    i = cuda.grid(1)
    n = pos.shape[0]

    if i >= n:  # Prevent OOB access
        return

    for j in range(i + 1, n):
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        dist_sq = dx * dx + dy * dy
        if dist_sq < cutoff_sq:
            # Atomic add to increment pair_count safely in parallel
            idx = cuda.atomic.add(pair_count, 0, 1)
            if idx < pairs.shape[0]: # Prevent out-of-bounds write for pairs array
                pairs[idx, 0] = i
                pairs[idx, 1] = j

# === Launch GPU kernel with timing ===
threads_per_block = 128
blocks = (N + threads_per_block - 1) // threads_per_block

start_gpu_kernel = time.perf_counter() # Use perf_counter for more precision
cuda_event_start = cuda.event()
cuda_event_end = cuda.event()
cuda_event_start.record()

compute_pairs[blocks, threads_per_block](d_positions, CUTOFF_SQ, d_pairs, d_pair_count)

cuda_event_end.record()
cuda_event_end.synchronize()
elapsed_gpu_kernel_ms = cuda.event_elapsed_time(cuda_event_start, cuda_event_end)
print(f"⏱ GPU kernel computation time: {elapsed_gpu_kernel_ms:.2f} ms")

# === Copy results back ===
start_copy_back = time.perf_counter()
pair_count = d_pair_count.copy_to_host()[0]
pairs = d_pairs.copy_to_host()[:pair_count]
positions_kpc = positions / 3.086e16  # Convert to kiloparsecs
end_copy_back = time.perf_counter()
print(f"⏱ Data copy back to CPU time: {(end_copy_back - start_copy_back) * 1000:.2f} ms")

print(f"✅ Total geodesic overlap lines identified: {pair_count}")

# === Plot (conditional for speed) ===
if not SKIP_PLOTTING:
    start_plotting = time.perf_counter()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('black')

    # Reduce alpha even further or skip drawing some lines if N is huge
    # Current alpha (0.11) is already good for N=20000, for N=5000 it's fine.
    for i, j in pairs:
        ax.plot([positions_kpc[i, 0], positions_kpc[j, 0]],
                [positions_kpc[i, 1], positions_kpc[j, 1]],
                color='cyan', alpha=0.11, linewidth=0.7, zorder=1)

    # Draw stars
    ax.scatter(positions_kpc[:, 0], positions_kpc[:, 1],
               color='magenta', s=15, alpha=0.92, label='Stars', zorder=2) # Reduced marker size

    # Draw CBH
    ax.scatter(0, 0, color='yellow', marker='*', s=180, label='CBH', zorder=3)

    ax.set_xlabel('X (kpc)')
    ax.set_ylabel('Y (kpc)')
    ax.set_title(f'Geodesic Overlap Web (CUDA Accelerated)\nN={N}, cutoff ~160 kpc')
    ax.legend(facecolor='black', framealpha=1, loc='lower left', fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()

    if SAVE_PLOT_TO_FILE:
        plt.savefig('geodesic_visualisation_fast.png', dpi=150, bbox_inches='tight')
        print("✅ Plot saved to 'geodesic_visualisation_fast.png'")

    plt.show()
    end_plotting = time.perf_counter()
    print(f"⏱ Plotting and display time: {(end_plotting - start_plotting) * 1000:.2f} ms")

else:
    print("📈 Plotting skipped for maximum speed.")
    if SAVE_PLOT_TO_FILE:
        # If not plotting, we still need to generate the figure if saving
        start_plotting = time.perf_counter()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        for i, j in pairs:
            ax.plot([positions_kpc[i, 0], positions_kpc[j, 0]],
                    [positions_kpc[i, 1], positions_kpc[j, 1]],
                    color='cyan', alpha=0.11, linewidth=0.7, zorder=1)
        ax.scatter(positions_kpc[:, 0], positions_kpc[:, 1],
                   color='magenta', s=15, alpha=0.92, label='Stars', zorder=2)
        ax.scatter(0, 0, color='yellow', marker='*', s=180, label='CBH', zorder=3)
        ax.set_xlabel('X (kpc)')
        ax.set_ylabel('Y (kpc)')
        ax.set_title(f'Geodesic Overlap Web (CUDA Accelerated)\nN={N}, cutoff ~160 kpc')
        ax.legend(facecolor='black', framealpha=1, loc='lower left', fontsize=10)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('geodesic_visualisation_fast.png', dpi=150, bbox_inches='tight')
        print("✅ Plot saved to 'geodesic_visualisation_fast.png' even though display was skipped.")
        plt.close(fig) # Close the figure to free memory if not displayed
        end_plotting = time.perf_counter()
        print(f"⏱ Plot saving (without display) time: {(end_plotting - start_plotting) * 1000:.2f} ms")

# Total execution time
total_script_time = time.perf_counter() - start_gpu_kernel # Start from when GPU work begins
print(f"\n⚡ Total script execution time (excluding initial setup): {total_script_time:.2f} seconds")
