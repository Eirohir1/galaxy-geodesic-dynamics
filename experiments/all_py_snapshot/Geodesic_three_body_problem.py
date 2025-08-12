import numpy as np
import cupy as cp
from numba import cuda, float64
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === Constants and parameters ===
N_MASSES = 3
masses_host = np.array([9.0, 6.0, 3.0], dtype=np.float64)
positions_host = np.array([
    [-10.0, 0.0],
    [10.0, 0.0],
    [0.0, 15.0],
], dtype=np.float64)

# Transfer to GPU
masses = cp.asarray(masses_host)
positions = cp.asarray(positions_host)

# === GPU device functions ===

@cuda.jit(device=True)
def schwarzschild_metric_00(mass, rx, ry):
    r = math.sqrt(rx*rx + ry*ry)
    if r < 1e-4:
        r = 1e-4
    return -(1 - 2*mass / r)

@cuda.jit(device=True)
def schwarzschild_metric_ii(mass, rx, ry):
    r = math.sqrt(rx*rx + ry*ry)
    if r < 1e-4:
        r = 1e-4
    return 1 + 2*mass / r

@cuda.jit(device=True)
def combined_metric(t, x, y, masses, positions, g):
    for i in range(4):
        for j in range(4):
            g[i,j] = 0.0
    g[0,0] = -1.0
    g[1,1] = 1.0
    g[2,2] = 1.0
    g[3,3] = 1.0

    for idx in range(N_MASSES):
        mass = masses[idx]
        px = positions[idx,0]
        py = positions[idx,1]
        rx = x - px
        ry = y - py
        g[0,0] += schwarzschild_metric_00(mass, rx, ry) + 1.0
        g[1,1] += schwarzschild_metric_ii(mass, rx, ry) - 1.0
        g[2,2] += schwarzschild_metric_ii(mass, rx, ry) - 1.0
        g[3,3] += schwarzschild_metric_ii(mass, rx, ry) - 1.0

@cuda.jit(device=True)
def invert_4x4_matrix(m, inv):
    # This is a simplified approximate inversion for demo purposes.
    # Replace with robust inversion in production.
    for i in range(4):
        for j in range(4):
            inv[i,j] = 0.0
    for i in range(4):
        inv[i,i] = 1.0

@cuda.jit(device=True)
def compute_christoffel(t, x, y, masses, positions, Gamma):
    h = 1e-5
    g = cuda.local.array((4,4), dtype=float64)
    g_p = cuda.local.array((4,4), dtype=float64)
    g_m = cuda.local.array((4,4), dtype=float64)
    dg = cuda.local.array((4,4,4), dtype=float64)

    combined_metric(t, x, y, masses, positions, g)
    g_inv = cuda.local.array((4,4), dtype=float64)
    invert_4x4_matrix(g, g_inv)

    # Numerical derivatives wrt coords
    for sigma in range(4):
        dt = t; dx = x; dy = y
        if sigma == 0:
            dt += h
            combined_metric(dt, x, y, masses, positions, g_p)
            dt -= 2*h
            combined_metric(dt, x, y, masses, positions, g_m)
        elif sigma == 1:
            dx += h
            combined_metric(t, dx, y, masses, positions, g_p)
            dx -= 2*h
            combined_metric(t, dx, y, masses, positions, g_m)
        elif sigma == 2:
            dy += h
            combined_metric(t, x, dy, masses, positions, g_p)
            dy -= 2*h
            combined_metric(t, x, dy, masses, positions, g_m)
        else:
            for i in range(4):
                for j in range(4):
                    dg[i,j,sigma] = 0.0
            continue
        for i in range(4):
            for j in range(4):
                dg[i,j,sigma] = (g_p[i,j] - g_m[i,j]) / (2*h)

    # Christoffel symbols calculation
    for mu in range(4):
        for alpha in range(4):
            for beta in range(alpha,4):
                val = 0.0
                for nu in range(4):
                    val += 0.5 * g_inv[mu,nu] * (dg[nu,beta,alpha] + dg[nu,alpha,beta] - dg[alpha,beta,nu])
                Gamma[mu,alpha,beta] = val
                Gamma[mu,beta,alpha] = val

@cuda.jit
def geodesic_kernel(tau_start, tau_end, dtau, y0, masses, positions, out_positions, out_velocities, out_count):
    y = cuda.local.array(8, dtype=float64)
    k1 = cuda.local.array(8, dtype=float64)
    k2 = cuda.local.array(8, dtype=float64)
    k3 = cuda.local.array(8, dtype=float64)
    k4 = cuda.local.array(8, dtype=float64)
    y_temp = cuda.local.array(8, dtype=float64)
    Gamma = cuda.local.array((4,4,4), dtype=float64)

    for i in range(8):
        y[i] = y0[i]

    steps = int((tau_end - tau_start) / dtau)
    count = 0

    def geodesic_derivative(y):
        coords = y[0:4]
        vel = y[4:8]
        t = coords[0]
        x = coords[1]
        y_ = coords[2]
        z = coords[3]

        compute_christoffel(t, x, y_, masses, positions, Gamma)

        acc = cuda.local.array(4, dtype=float64)
        for mu in range(4):
            acc[mu] = 0.0
            for alpha in range(4):
                for beta in range(4):
                    acc[mu] -= Gamma[mu,alpha,beta] * vel[alpha] * vel[beta]

        dy_out = cuda.local.array(8, dtype=float64)
        for i in range(4):
            dy_out[i] = vel[i]
            dy_out[i+4] = acc[i]
        return dy_out

    for step in range(steps):
        dy1 = geodesic_derivative(y)
        for i in range(8):
            k1[i] = dtau * dy1[i]

        for i in range(8):
            y_temp[i] = y[i] + 0.5 * k1[i]
        dy2 = geodesic_derivative(y_temp)
        for i in range(8):
            k2[i] = dtau * dy2[i]

        for i in range(8):
            y_temp[i] = y[i] + 0.5 * k2[i]
        dy3 = geodesic_derivative(y_temp)
        for i in range(8):
            k3[i] = dtau * dy3[i]

        for i in range(8):
            y_temp[i] = y[i] + k3[i]
        dy4 = geodesic_derivative(y_temp)
        for i in range(8):
            k4[i] = dtau * dy4[i]

        for i in range(8):
            y[i] = y[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0

        if count < out_positions.shape[0]:
            out_positions[count, 0] = y[1]
            out_positions[count, 1] = y[2]
            out_velocities[count, 0] = y[5]
            out_velocities[count, 1] = y[6]
            count += 1

    out_count[0] = count

# === Host wrapper to run kernel ===

def run_gpu_geodesic():
    dtau = 0.01
    tau_start = 0.0
    tau_end = 20.0  # shorter for demo
    steps = int((tau_end - tau_start) / dtau)

    y0 = np.zeros(8, dtype=np.float64)
    y0[0] = 0.0  # t
    y0[1] = 5.0  # x initial
    y0[2] = 5.0  # y initial
    y0[3] = 0.0  # z initial
    y0[4] = 1.0  # dt/dtau
    y0[5] = 0.0  # vx
    y0[6] = 0.5  # vy
    y0[7] = 0.0  # vz

    d_y0 = cp.asarray(y0)
    d_positions = cp.zeros((steps, 2), dtype=cp.float64)
    d_velocities = cp.zeros((steps, 2), dtype=cp.float64)
    d_step_count = cp.zeros(1, dtype=cp.int32)

    # Launch kernel with single thread (simulate one geodesic)
    geodesic_kernel[1, 1](tau_start, tau_end, dtau, d_y0, masses, positions, d_positions, d_velocities, d_step_count)

    cp.cuda.Stream.null.synchronize()

    count = int(d_step_count.get()[0])
    x_arr = d_positions.get()[:count, 0]
    y_arr = d_positions.get()[:count, 1]
    vx_arr = d_velocities.get()[:count, 0]
    vy_arr = d_velocities.get()[:count, 1]

    return x_arr, y_arr, vx_arr, vy_arr

# === Radial velocity profile from trajectory ===

def compute_radial_velocity_profile(x_arr, y_arr, vx_arr, vy_arr, num_bins=50):
    r = np.sqrt(x_arr**2 + y_arr**2)
    v_tangential = (x_arr * vy_arr - y_arr * vx_arr) / r  # cross product magnitude / radius
    bins = np.linspace(np.min(r), np.max(r), num_bins+1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    v_profile = np.zeros(num_bins)
    counts = np.zeros(num_bins)
    for i in range(len(r)):
        bin_idx = np.searchsorted(bins, r[i]) - 1
        if 0 <= bin_idx < num_bins:
            v_profile[bin_idx] += v_tangential[i]
            counts[bin_idx] += 1
    v_profile = np.divide(v_profile, counts, out=np.zeros_like(v_profile), where=counts>0)
    return bin_centers, v_profile

# === Kernel functions and fitting from your kernel_sensitivity_analysis.py ===

def exponential_kernel(r, ell):
    return np.exp(-r / ell)

def v_baryonic(r_kpc, data):
    # For this example, assume zero baryonic velocity (simulation only)
    return np.zeros_like(r_kpc)

def v_dark_general(r_kpc, data, ell, alpha, g_inf, kernel_func):
    # Simple placeholder to fit
    kern = kernel_func(r_kpc, ell)
    return alpha * kern + g_inf

def v_total_general(r_kpc, data, ell, alpha, g_inf, kernel_func):
    v_bar = v_baryonic(r_kpc, data)
    v_dm = v_dark_general(r_kpc, data, ell, alpha, g_inf, kernel_func)
    return np.sqrt(v_bar**2 + v_dm**2)

def fit_galaxy_with_kernel(data, kernel_func, kernel_name="unknown"):
    ell_range = (0.5, 35.0)
    alpha_range = (0.01, 1.0)
    g_inf_range = (0.0, 0.6)

    def objective(params):
        ell, alpha, g_inf = params
        if not (ell_range[0] <= ell <= ell_range[1]): return 1e8
        if not (alpha_range[0] <= alpha <= alpha_range[1]): return 1e8
        if not (g_inf_range[0] <= g_inf <= g_inf_range[1]): return 1e8
        v_pred = v_total_general(data.r_kpc, data, ell, alpha, g_inf, kernel_func)
        if np.any(~np.isfinite(v_pred)):
            return 1e8
        chi2 = np.sum(((data.v_obs - v_pred) / data.dv_obs)**2)
        return chi2

    from scipy.optimize import minimize
    attempts = [(5.0, 0.15, 0.05), (2.0, 0.3, 0.1), (10.0, 0.1, 0.02), (15.0, 0.5, 0.0)]

    best_params = None
    best_chi2 = 1e8
    for start in attempts:
        try:
            result = minimize(objective, start, method='Nelder-Mead',
                              options={'maxiter':2000, 'fatol':1e-8})
            if result.fun < best_chi2:
                best_chi2 = result.fun
                best_params = result.x
        except:
            continue

    return best_params, best_chi2

# === Main ===

def main():
    print("Running GPU geodesic simulation...")
    x_arr, y_arr, vx_arr, vy_arr = run_gpu_geodesic()

    print("Computing radial velocity profile...")
    radii, v_profile = compute_radial_velocity_profile(x_arr, y_arr, vx_arr, vy_arr)

    # Prepare dummy data class for fitting
    class DummyData:
        def __init__(self, r, v):
            self.r_kpc = r
            self.v_obs = v
            self.dv_obs = np.full_like(v, 0.05)

    sim_data = DummyData(radii, v_profile)

    print("Fitting kernel to simulated rotation curve...")
    params, chi2 = fit_galaxy_with_kernel(sim_data, exponential_kernel, "Exponential")

    print(f"Fit results: ell={params[0]:.3f}, alpha={params[1]:.3f}, g_inf={params[2]:.3f}, chi2={chi2:.2f}")

    v_fit = v_total_general(sim_data.r_kpc, sim_data, *params, exponential_kernel)

    plt.figure(figsize=(8,6))
    plt.plot(sim_data.r_kpc, sim_data.v_obs, 'ko', label="Simulated data")
    plt.plot(sim_data.r_kpc, v_fit, 'r-', label="Exponential Kernel Fit")
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.legend()
    plt.title("Kernel Fit to Simulated Rotation Curve")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
