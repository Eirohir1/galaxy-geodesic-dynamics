import numpy as np
from numba import jit, prange
import time

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
# CPU-optimized force calculations
# ========================
@jit(nopython=True, parallel=True)
def add_blackhole_force(x, y, z, ax, ay, az):
    """Add central black hole gravitational force into ax,ay,az (accumulate)."""
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
    """
    Standard Newtonian N-body forces.
    Writes the NEW contribution into ax,ay,az (overwrites their content).
    """
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
def compute_geodesic_forces(x, y, z, masses, ax, ay, az):
    """
    Geodesic reinforcement forces (Gaussian-weighted nonlocal term).
    Writes the NEW contribution into ax,ay,az (overwrites their content).
    """
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

@jit(nopython=True, parallel=True)
def leapfrog_halfkick_pos_fullkick(x, y, z, vx, vy, vz, ax, ay, az, dt):
    """
    Do a leapfrog step in-place given accelerations ax,ay,az:
      1) half-kick: v += 0.5*a*dt
      2) drift:    x += v*dt
    The second half-kick is done by calling this again after
    accelerations are recomputed, OR by separate helper.
    (We’ll call this helper twice per full step with recomputed a’s.)
    """
    N = len(x)
    hdt = 0.5 * dt
    for i in prange(N):
        vx[i] += hdt * ax[i]
        vy[i] += hdt * ay[i]
        vz[i] += hdt * az[i]
        x[i]  += dt * vx[i]
        y[i]  += dt * vy[i]
        z[i]  += dt * vz[i]

@jit(nopython=True, parallel=True)
def vel_halfkick(vx, vy, vz, ax, ay, az, dt):
    """Apply the second half-kick: v += 0.5*a*dt."""
    N = len(vx)
    hdt = 0.5 * dt
    for i in prange(N):
        vx[i] += hdt * ax[i]
        vy[i] += hdt * ay[i]
        vz[i] += hdt * az[i]

# ========================
# Simulation function
# ========================
def run_simulation(include_newton: bool, include_geodesic: bool, label: str):
    """
    Runs one simulation with the chosen force terms.
    Forces included:
      - Always: central BH
      - Optional: Newton p–p
      - Optional: Geodesic reinforcement
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

        if include_newton:
            compute_newton_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz

        if include_geodesic:
            compute_geodesic_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz

        # Half-kick + drift
        leapfrog_halfkick_pos_fullkick(x_sim, y_sim, z_sim, vx_sim, vy_sim, vz_sim, ax, ay, az, dt)

        # ---- Second force eval (for second half-kick) ----
        ax.fill(0.0); ay.fill(0.0); az.fill(0.0)
        add_blackhole_force(x_sim, y_sim, z_sim, ax, ay, az)

        if include_newton:
            compute_newton_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz

        if include_geodesic:
            compute_geodesic_forces(x_sim, y_sim, z_sim, masses, tmpx, tmpy, tmpz)
            ax += tmpx; ay += tmpy; az += tmpz

        # Second half-kick
        vel_halfkick(vx_sim, vy_sim, vz_sim, ax, ay, az, dt)

        # Progress
        if step % 10000 == 0:
            print(f"{label} step {step}/{n_steps}")

    runtime = time.time() - start_time

    # Analysis (bound to CBH only, for consistency with earlier runs)
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
    print(f"Energy range: [{np.min(E):.6f}, {np.max(E):.6f}]")

    return {
        "label": label,
        "runtime": runtime,
        "bound": bound_frac,
        "maxR": float(np.max(r)),
        "meanR": float(np.mean(r))
    }

# ========================
# Run three simulations and compare
# ========================
if __name__ == "__main__":
    print("=== N-Body Simulation: Newton vs (Newton+Geo) vs Geo-only ===")
    print(f"N={N} particles, {n_steps} steps, dt={dt}")
    print(f"Geodesic params: alpha={alpha}, ell={ell}")
    print("=" * 60)

    # Case A: CBH + Newton
    caseA = run_simulation(include_newton=True,  include_geodesic=False, label="Case A: Newton")

    print("\n" + "=" * 60)

    # Case B: CBH + Newton + Geodesic (key apples-to-apples test)
    caseB = run_simulation(include_newton=True,  include_geodesic=True,  label="Case B: Newton + Geodesic")

    print("\n" + "=" * 60)

    # Case C: CBH + Geodesic only (what you ran before)
    caseC = run_simulation(include_newton=False, include_geodesic=True,  label="Case C: Geodesic-only")

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY:")
    print(f"A) Newton:             bound = {caseA['bound']:.3f}, runtime = {caseA['runtime']:.1f}s, meanR = {caseA['meanR']:.2f}")
    print(f"B) Newton + Geodesic:  bound = {caseB['bound']:.3f}, runtime = {caseB['runtime']:.1f}s, meanR = {caseB['meanR']:.2f}")
    print(f"C) Geodesic-only:      bound = {caseC['bound']:.3f}, runtime = {caseC['runtime']:.1f}s, meanR = {caseC['meanR']:.2f}")

    dAB = (caseB['bound'] - caseA['bound']) * 100.0
    dAC = (caseC['bound'] - caseA['bound']) * 100.0
    print(f"\nIncremental retention (B - A): {dAB:+.2f} percentage points  <-- apples-to-apples")
    print(f"Difference (C - A):            {dAC:+.2f} percentage points  <-- reinforcement vs Newton (no Newton p–p)")
    print("Simulation complete!")
