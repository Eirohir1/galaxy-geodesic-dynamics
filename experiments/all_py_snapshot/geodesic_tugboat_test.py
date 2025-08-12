import numpy as np
from numba import jit, prange
import time
import matplotlib.pyplot as plt

# ========================
# Parameters
# ========================
N = 500
G = 1.0
M_bh = 1.0
alpha = 0.3
ell = 0.2
eps = 0.02
dt = 0.002
n_steps = 50000
R_d = 0.5
R_max = 3.0
vel_disp_frac = 0.05

# Tug boat parameters
beta = 0.5  # Strength of radial momentum transfer
gamma = 2.0  # Velocity differential amplification

# ========================
# Initial conditions
# ========================
def sample_exponential_disk(N, R_d, R_max):
    radii = []
    while len(radii) < N:
        r = -R_d * np.log(1 - np.random.rand())
        if r <= R_max:
            radii.append(r)
    radii = np.array(radii)
    theta = np.random.rand(N) * 2 * np.pi
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return x, y

x, y = sample_exponential_disk(N, R_d, R_max)
m_part = 1.0 / N
masses = np.full(N, m_part)
z = np.zeros(N)

R = np.sqrt(x**2 + y**2)
v_circ = np.sqrt(G * M_bh / R)
vx = -v_circ * np.sin(np.arctan2(y, x))
vy =  v_circ * np.cos(np.arctan2(y, x))
vx += np.random.randn(N) * vel_disp_frac * v_circ
vy += np.random.randn(N) * vel_disp_frac * v_circ
vz = np.zeros(N)

# Save initial conditions
x0, y0, z0 = x.copy(), y.copy(), z.copy()
vx0, vy0, vz0 = vx.copy(), vy.copy(), vz.copy()

# ========================
# Enhanced force calculations with tug boat mechanism
# ========================
@jit(nopython=True, parallel=True)
def add_blackhole_force(x, y, z, ax, ay, az):
    """Add central black hole gravitational force."""
    N = len(x)
    for i in prange(N):
        xi, yi, zi = x[i], y[i], z[i]
        r2 = xi*xi + yi*yi + zi*zi + eps*eps
        inv_r3 = 1.0 / (r2 * np.sqrt(r2))
        ax[i] += -G * M_bh * xi * inv_r3
        ay[i] += -G * M_bh * yi * inv_r3
        az[i] += -G * M_bh * zi * inv_r3

@jit(nopython=True, parallel=True)
def compute_newton_forces(x, y, z, masses, ax, ay, az):
    """Standard Newtonian N-body forces."""
    N = len(x)
    for i in prange(N):
        axi = 0.0
        ayi = 0.0
        azi = 0.0
        xi, yi, zi = x[i], y[i], z[i]
        for j in range(N):
            if i != j:
                dx = x[j] - xi
                dy = y[j] - yi
                dz = z[j] - zi
                r2 = dx*dx + dy*dy + dz*dz + eps*eps
                inv_r3 = 1.0 / (r2 * np.sqrt(r2))
                mj = masses[j]
                axi += G * mj * dx * inv_r3
                ayi += G * mj * dy * inv_r3
                azi += G * mj * dz * inv_r3
        ax[i] = axi
        ay[i] = ayi
        az[i] = azi

@jit(nopython=True, parallel=True)
def compute_tugboat_forces(x, y, z, vx, vy, vz, masses, ax, ay, az):
    """
    Geodesic tug boat mechanism: Inner fast stars boost outer slow stars.
    Key physics: Radial momentum transfer weighted by velocity difference.
    """
    N = len(x)
    for i in prange(N):
        axi = 0.0
        ayi = 0.0
        azi = 0.0
        xi, yi, zi = x[i], y[i], z[i]
        vxi, vyi, vzi = vx[i], vy[i], vz[i]
        ri = np.sqrt(xi*xi + yi*yi + zi*zi)
        v_i = np.sqrt(vxi*vxi + vyi*vyi + vzi*vzi)
        
        for j in range(N):
            if i != j:
                xj, yj, zj = x[j], y[j], z[j]
                vxj, vyj, vzj = vx[j], vy[j], vz[j]
                rj = np.sqrt(xj*xj + yj*yj + zj*zj)
                v_j = np.sqrt(vxj*vxj + vyj*vyj + vzj*vzj)
                
                # Separation and direction
                dx = xj - xi
                dy = yj - yi
                dz = zj - zi
                r_sep = np.sqrt(dx*dx + dy*dy + dz*dz + eps*eps)
                
                # Geodesic coupling weight
                w = np.exp(-0.5 * (r_sep/ell)**2)
                
                # Tug boat mechanism: 
                # - Inner (smaller r) high-v particles boost outer (larger r) low-v particles
                # - Force is radially outward for momentum transfer
                if rj < ri and v_j > v_i:  # j is inner and faster
                    # Velocity difference amplifies the effect
                    v_diff = v_j - v_i
                    transfer_strength = beta * alpha * w * v_diff**gamma
                    
                    # Radial direction from center to particle i (outward boost)
                    r_hat_x = xi / (ri + eps)
                    r_hat_y = yi / (ri + eps)
                    r_hat_z = zi / (ri + eps)
                    
                    # Apply outward radial force
                    force_mag = transfer_strength * masses[j] / (r_sep**2 + eps*eps)
                    axi += force_mag * r_hat_x
                    ayi += force_mag * r_hat_y
                    azi += force_mag * r_hat_z
        
        ax[i] = axi
        ay[i] = ayi
        az[i] = azi

@jit(nopython=True, parallel=True)
def compute_standard_geodesic_forces(x, y, z, masses, ax, ay, az):
    """Original geodesic reinforcement for comparison."""
    N = len(x)
    for i in prange(N):
        axi = 0.0
        ayi = 0.0
        azi = 0.0
        xi, yi, zi = x[i], y[i], z[i]
        for j in range(N):
            if i != j:
                dx = x[j] - xi
                dy = y[j] - yi
                dz = z[j] - zi
                r2 = dx*dx + dy*dy + dz*dz + eps*eps
                r = np.sqrt(r2)
                w = np.exp(-0.5 * (r/ell)**2)
                inv_r3 = 1.0 / (r2 * r)
                mj = masses[j]
                axi += alpha * G * mj * dx * inv_r3 * w
                ayi += alpha * G * mj * dy * inv_r3 * w
                azi += alpha * G * mj * dz * inv_r3 * w
        ax[i] = axi
        ay[i] = ayi
        az[i] = azi

# ========================
# Simulation function
# ========================
def run_simulation(force_type: str, label: str):
    """
    Run simulation with different force combinations:
    - 'newton': CBH + Newton p-p
    - 'standard_geo': CBH + standard geodesic
    - 'tugboat': CBH + Newton p-p + tug boat mechanism
    """
    # Reset to initial conditions
    x_sim = x0.copy()
    y_sim = y0.copy()
    z_sim = z0.copy()
    vx_sim = vx0.copy()
    vy_sim = vy0.copy()
    vz_sim = vz0.copy()

    # Acceleration work arrays
    ax = np.zeros(N)
    ay = np.zeros(N)
    az = np.zeros(N)
    tmpx = np.zeros(N)
    tmpy = np.zeros(N)
    tmpz = np.zeros(N)

    print(f"Starting {label} simulation...")
    start_time = time.time()

    for step in range(n_steps):
        # ---- First force eval (for half-kick + drift) ----
        ax.fill(0.0); ay.fill(0.0); az.fill(0.0)
        add_blackhole_force(x_sim, y_sim, z_sim, ax, ay, az)

        if force_type == 'newton':
            compute_newton_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz
        elif force_type == 'standard_geo':
            compute_standard_geodesic_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz
        elif force_type == 'tugboat':
            compute_newton_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz
            compute_tugboat_forces(x_sim, y_sim, z_sim, vx_sim, vy_sim, vz_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz

        # Half-kick + drift
        vx_sim += 0.5 * ax * dt
        vy_sim += 0.5 * ay * dt
        vz_sim += 0.5 * az * dt
        x_sim += vx_sim * dt
        y_sim += vy_sim * dt
        z_sim += vz_sim * dt

        # ---- Second force eval (for second half-kick) ----
        ax.fill(0.0); ay.fill(0.0); az.fill(0.0)
        add_blackhole_force(x_sim, y_sim, z_sim, ax, ay, az)

        if force_type == 'newton':
            compute_newton_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz
        elif force_type == 'standard_geo':
            compute_standard_geodesic_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz
        elif force_type == 'tugboat':
            compute_newton_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz
            compute_tugboat_forces(x_sim, y_sim, z_sim, vx_sim, vy_sim, vz_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz

        # Second half-kick
        vx_sim += 0.5 * ax * dt
        vy_sim += 0.5 * ay * dt
        vz_sim += 0.5 * az * dt

        # Progress
        if step % 10000 == 0:
            print(f"{label} step {step}/{n_steps}")

    runtime = time.time() - start_time

    # Analysis
    r = np.sqrt(x_sim**2 + y_sim**2 + z_sim**2)
    v2 = vx_sim**2 + vy_sim**2 + vz_sim**2
    phi_bh = -G * M_bh / np.sqrt(r**2 + eps**2)
    E = 0.5 * v2 + phi_bh
    bound_frac = np.sum(E < 0) / N

    print(f"\n{label} Results:")
    print(f"Runtime: {runtime:.2f}s")
    print(f"Bound fraction: {bound_frac:.3f}")
    print(f"Max radius: {np.max(r):.3f}")
    print(f"Mean radius: {np.mean(r):.3f}")

    # Calculate rotation curve
    r_bins = np.linspace(0.1, 3.0, 30)
    v_rot = []
    
    for r_bin in r_bins:
        mask = (r >= r_bin - 0.05) & (r <= r_bin + 0.05)
        if np.sum(mask) > 0:
            # Calculate circular velocity
            v_circ_avg = np.sqrt(np.mean(vx_sim[mask]**2 + vy_sim[mask]**2))
            v_rot.append(v_circ_avg)
        else:
            v_rot.append(0)
    
    return {
        "label": label,
        "runtime": runtime,
        "bound": bound_frac,
        "r_curve": r_bins,
        "v_curve": np.array(v_rot)
    }

# ========================
# Run simulations and compare
# ========================
if __name__ == "__main__":
    print("=== Testing Geodesic Tug Boat Mechanism ===")
    print(f"N={N} particles, {n_steps} steps")
    print(f"Tug boat params: beta={beta}, gamma={gamma}")
    print("=" * 60)

    # Run three cases
    newton_result = run_simulation('newton', 'Newton Baseline')
    print("\n" + "=" * 60)
    
    standard_result = run_simulation('standard_geo', 'Standard Geodesic')
    print("\n" + "=" * 60)
    
    tugboat_result = run_simulation('tugboat', 'Tug Boat Mechanism')
    
    print("\n" + "=" * 60)
    print("ROTATION CURVE COMPARISON:")
    print(f"Newton:     bound = {newton_result['bound']:.3f}")
    print(f"Standard:   bound = {standard_result['bound']:.3f}")  
    print(f"Tug Boat:   bound = {tugboat_result['bound']:.3f}")
    
    # Plot rotation curves
    plt.figure(figsize=(10, 6))
    plt.plot(newton_result['r_curve'], newton_result['v_curve'], 'b-', label='Newton', linewidth=2)
    plt.plot(standard_result['r_curve'], standard_result['v_curve'], 'r-', label='Standard Geodesic', linewidth=2)
    plt.plot(tugboat_result['r_curve'], tugboat_result['v_curve'], 'g-', label='Tug Boat Mechanism', linewidth=2)
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title('Rotation Curves: Testing Tug Boat Mechanism')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Check if tug boat creates rising curves
    tb_slope = (tugboat_result['v_curve'][-1] - tugboat_result['v_curve'][5]) / (tugboat_result['r_curve'][-1] - tugboat_result['r_curve'][5])
    newton_slope = (newton_result['v_curve'][-1] - newton_result['v_curve'][5]) / (newton_result['r_curve'][-1] - newton_result['r_curve'][5])
    
    print(f"\nCurve slopes (outer region):")
    print(f"Newton slope: {newton_slope:.3f}")
    print(f"Tug boat slope: {tb_slope:.3f}")
    
    if tb_slope > newton_slope:
        print("🎯 SUCCESS: Tug boat mechanism creates flatter/rising curves!")
    else:
        print("❌ Tug boat mechanism still declining - need parameter adjustment")
    
    print("Simulation complete!")