# NewtonvsGeodesics_v11.py
# GPU-accelerated realistic galaxy + head-to-head Newton vs Geodesic (neighbor-overlap) test
# - Mode "newton": CBH + bulge + disk (no dark-matter flat term)
# - Mode "geodesic": same + geodesic overlap kernel (local neighbor Venn-slice acceleration)
#
# Outputs:
#   - gpu_galaxy_structure_{MODE}.png  (diagnostic 6-panel plot)
#   - escapes_{MODE}.csv               (per-star escape status & initial radius/region)
#   - Console summary with total escapes + by-region breakdown
#
# Sensible defaults chosen to match your description. No tuning required to run.

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# -----------------------------
# GPU setup
# -----------------------------
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU ACCELERATION ENABLED!")
    try:
        gpu_device = cp.cuda.Device()
        mem_free, mem_total = gpu_device.mem_info
        print(f"GPU Memory: {mem_free/1e9:.1f} GB available")
    except Exception:
        pass
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("⚠️ CuPy not found — running on CPU (slower). Install a matching CuPy build for your CUDA.")

# -----------------------------
# Galaxy model class
# -----------------------------
class RealisticGalaxy:
    """
    Realistic Milky Way–like galaxy:
      - Central black hole (CBH)
      - Hernquist bulge (spheroidal)
      - Exponential disk (thin, with bar option)
      - Logarithmic spiral arms (younger population)
      - Velocity dispersions & streaming
      - GPU-native data storage (CuPy) when available
    """
    def __init__(self,
                 num_stars=10000,
                 cbh_mass=4.15e6,     # Sgr A* (M_sun)
                 bulge_mass=2e10,     # MW bulge (M_sun)
                 disk_mass=6e10,      # MW disk (M_sun)
                 galaxy_radius=20.0,  # kpc
                 scale_height=0.3,    # kpc
                 num_arms=4,
                 arm_pitch_deg=12.0,
                 bar_strength=0.3,
                 use_gpu=True,
                 rng_seed=42):
        self.G = 4.30091e-6  # gravitational constant in (kpc * (km/s)^2) / M_sun
        self.num_stars = int(num_stars)
        self.cbh_mass = cbh_mass
        self.bulge_mass = bulge_mass
        self.disk_mass = disk_mass
        self.galaxy_radius = galaxy_radius
        self.scale_height = scale_height
        self.num_arms = num_arms
        self.arm_pitch = np.deg2rad(arm_pitch_deg)
        self.bar_strength = bar_strength
        self.use_gpu = (use_gpu and GPU_AVAILABLE)
        self.rng = np.random.default_rng(rng_seed)

        # Structural scales
        self.bulge_radius = galaxy_radius * 0.2
        self.disk_scale_length = galaxy_radius / 3.5

        self._create_structure()
        self._calc_stats()

    # --------- builders ----------
    def _create_structure(self):
        print("\n" + "="*60)
        print("BUILDING GPU-ACCELERATED GALAXY MODEL")
        print("="*60)

        n_bulge = int(0.25 * self.num_stars)
        n_disk  = int(0.60 * self.num_stars)
        n_arms  = self.num_stars - n_bulge - n_disk

        print("Stellar distribution:")
        print(f"  Central bulge: {n_bulge} stars")
        print(f"  Disk: {n_disk} stars")
        print(f"  Spiral arms: {n_arms} stars")
        print(f"  Using GPU: {self.use_gpu}")

        bp, bv, bm = self._create_bulge(n_bulge)
        dp, dv, dm = self._create_disk(n_disk)
        ap, av, am = self._create_spiral_arms(n_arms)

        pos_cpu = np.vstack([bp, dp, ap])
        vel_cpu = np.vstack([bv, dv, av])
        mass_cpu = np.hstack([bm, dm, am])

        self.component_labels = np.array(['bulge']*n_bulge + ['disk']*n_disk + ['arms']*n_arms)

        if self.use_gpu:
            self.positions = cp.asarray(pos_cpu)
            self.velocities = cp.asarray(vel_cpu)
            self.masses    = cp.asarray(mass_cpu)

            self.x = self.positions[:,0]; self.y = self.positions[:,1]; self.z = self.positions[:,2]
            self.vx = self.velocities[:,0]; self.vy = self.velocities[:,1]; self.vz = self.velocities[:,2]
        else:
            self.positions = pos_cpu
            self.velocities = vel_cpu
            self.masses    = mass_cpu

            self.x = self.positions[:,0]; self.y = self.positions[:,1]; self.z = self.positions[:,2]
            self.vx = self.velocities[:,0]; self.vy = self.velocities[:,1]; self.vz = self.velocities[:,2]

        # for escape stats later
        self.initial_radius = self._norm_r(self.x, self.y)

    def _create_bulge(self, n):
        print("Creating central bulge...")
        a = self.bulge_radius * 0.5
        u = self.rng.uniform(0, 1, n)
        r = a * np.sqrt(u) / (1 - np.sqrt(u))
        r = np.clip(r, 0, self.bulge_radius * 2)

        theta = self.rng.uniform(0, 2*np.pi, n)
        phi = np.arccos(1 - 2*self.rng.uniform(0, 1, n))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) * 0.6
        pos = np.column_stack([x, y, z])

        masses = np.clip(self.rng.lognormal(0.3, 0.5, n), 0.3, 10.0)

        r_cyl = np.sqrt(x**2 + y**2)
        v_rot = self._rotation_curve_scalar(r_cyl)

        sigma = 100.0
        v_disp_r = self.rng.normal(0, sigma, n)
        v_disp_t = self.rng.normal(0, sigma, n)
        v_disp_z = self.rng.normal(0, 0.7*sigma, n)

        v_theta = v_rot + v_disp_t
        vx = v_disp_r * np.cos(theta) - v_theta * np.sin(theta)
        vy = v_disp_r * np.sin(theta) + v_theta * np.cos(theta)
        vz = v_disp_z
        vel = np.column_stack([vx, vy, vz])
        return pos, vel, masses

    def _create_disk(self, n):
        print("Creating exponential disk...")
        r = self.rng.exponential(self.disk_scale_length, n)
        r = np.clip(r, self.bulge_radius, self.galaxy_radius)
        theta = self.rng.uniform(0, 2*np.pi, n)
        # vertical: approximate symmetric sech^2 via clipped tanh inverse sampling
        z = self.scale_height * np.arctanh(self.rng.uniform(-0.98, 0.98, n))

        if self.bar_strength > 0:
            bar_angle = np.pi/4
            r_bar = r * (1 + self.bar_strength * np.cos(2*(theta - bar_angle)))
            x = r_bar * np.cos(theta)
            y = r_bar * np.sin(theta)
        else:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        pos = np.column_stack([x, y, z])

        masses = self._sample_imf(n, young=False)

        v_circ = self._rotation_curve_scalar(r)
        sigma_r = 30.0 * np.exp(-r / (2*self.disk_scale_length))
        sigma_t = 0.7 * sigma_r
        sigma_z = 0.5 * sigma_r

        v_r = self.rng.normal(0, sigma_r, n)
        v_t = v_circ + self.rng.normal(0, sigma_t, n)
        v_z = self.rng.normal(0, sigma_z, n)

        vx = v_r * np.cos(theta) - v_t * np.sin(theta)
        vy = v_r * np.sin(theta) + v_t * np.cos(theta)
        vz = v_z
        vel = np.column_stack([vx, vy, vz])
        return pos, vel, masses

    def _create_spiral_arms(self, n):
        print("Creating spiral arms...")
        if n <= 0:
            return (np.zeros((0,3)), np.zeros((0,3)), np.zeros((0,)))
        r = self.rng.exponential(self.disk_scale_length*1.2, n)
        r = np.clip(r, self.bulge_radius*1.5, self.galaxy_radius)
        arm_idx = self.rng.integers(0, self.num_arms, n)

        b = np.tan(self.arm_pitch)
        a0 = self.bulge_radius
        theta = np.log(r / a0) / b + (2*np.pi * arm_idx / self.num_arms) + self.rng.normal(0, 0.1, n)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = self.scale_height * self.rng.normal(0, 0.5, n)
        pos = np.column_stack([x, y, z])

        masses = self._sample_imf(n, young=True)

        v_circ = self._rotation_curve_scalar(r)
        v_stream = 10.0
        v_r = self.rng.normal(v_stream, 20.0, n)
        v_t = v_circ + self.rng.normal(0, 25.0, n)
        v_z = self.rng.normal(0, 15.0, n)

        vx = v_r * np.cos(theta) - v_t * np.sin(theta)
        vy = v_r * np.sin(theta) + v_t * np.cos(theta)
        vz = v_z
        vel = np.column_stack([vx, vy, vz])
        return pos, vel, masses

    def _sample_imf(self, n, young=False):
        if young:
            m = np.clip(self.rng.lognormal(0.5, 0.7, n), 0.5, 50.0)
        else:
            # simple Kroupa-like piecewise
            out = np.empty(n, dtype=float)
            u = self.rng.uniform(0, 1, n)
            mask1 = (u < 0.5)
            mask2 = (u >= 0.5) & (u < 0.9)
            mask3 = (u >= 0.9)
            out[mask1] = self.rng.uniform(0.08, 0.5, np.count_nonzero(mask1))
            out[mask2] = self.rng.uniform(0.5, 1.0, np.count_nonzero(mask2))
            out[mask3] = self.rng.uniform(1.0, 8.0, np.count_nonzero(mask3))
            m = out
        return m

    # --------- physics helpers ----------
    def _norm_r(self, x, y):
        if self.use_gpu:
            return cp.sqrt(x*x + y*y)
        else:
            return np.sqrt(x*x + y*y)

    def _rotation_curve_scalar(self, r):
        # r is numpy array, returns numpy array (used during build)
        eps = 1e-3
        r = np.maximum(r, eps)
        # CBH
        v_cbh = np.sqrt(self.G * self.cbh_mass / (r + 0.001))
        # Hernquist bulge
        a = self.bulge_radius * 0.5
        m_enc_b = self.bulge_mass * r*r / (r + a)**2
        v_bulge = np.sqrt(self.G * m_enc_b / r)
        # Exponential disk
        m_enc_d = self.disk_mass * (1 - (1 + r/self.disk_scale_length) * np.exp(-r/self.disk_scale_length))
        v_disk = np.sqrt(self.G * m_enc_d / (r + 0.1))
        # Combine — NO dark matter term
        v_tot = np.sqrt(v_cbh*v_cbh + v_bulge*v_bulge + v_disk*v_disk)
        return v_tot

    def _rotation_curve_gpu(self, r_gpu):
        # r_gpu is cupy or numpy depending on use_gpu
        eps = 1e-3
        r = cp.maximum(r_gpu, eps) if self.use_gpu else np.maximum(r_gpu, eps)
        # CBH
        v_cbh = cp.sqrt(self.G * self.cbh_mass / (r + 0.001)) if self.use_gpu else np.sqrt(self.G * self.cbh_mass / (r + 0.001))
        # Bulge
        a = self.bulge_radius * 0.5
        m_enc_b = self.bulge_mass * r*r / (r + a)**2
        v_bulge = cp.sqrt(self.G * m_enc_b / r) if self.use_gpu else np.sqrt(self.G * m_enc_b / r)
        # Disk
        m_enc_d = self.disk_mass * (1 - (1 + r/self.disk_scale_length) * (cp.exp(-r/self.disk_scale_length) if self.use_gpu else np.exp(-r/self.disk_scale_length)))
        v_disk = cp.sqrt(self.G * m_enc_d / (r + 0.1)) if self.use_gpu else np.sqrt(self.G * m_enc_d / (r + 0.1))
        v_tot = cp.sqrt(v_cbh*v_cbh + v_bulge*v_bulge + v_disk*v_disk) if self.use_gpu else np.sqrt(v_cbh*v_cbh + v_bulge*v_bulge + v_disk*v_disk)
        return v_tot

    # --------- stats & plotting ----------
    def _calc_stats(self):
        print("\n" + "="*60)
        print("GALAXY STATISTICS")
        print("="*60)
        if self.use_gpu:
            x = cp.asnumpy(self.x); y = cp.asnumpy(self.y); z = cp.asnumpy(self.z)
            m = cp.asnumpy(self.masses)
            vx = cp.asnumpy(self.vx); vy = cp.asnumpy(self.vy); vz = cp.asnumpy(self.vz)
        else:
            x, y, z = self.x, self.y, self.z
            m = self.masses
            vx, vy, vz = self.vx, self.vy, self.vz

        r = np.sqrt(x*x + y*y)
        v = np.sqrt(vx*vx + vy*vy + vz*vz)

        print("\nMass distribution:")
        print(f"  Total stellar mass: {np.sum(m):.2e} M☉")
        print(f"  Average stellar mass: {np.mean(m):.2f} M☉")
        print(f"  Mass range: {np.min(m):.2f} - {np.max(m):.2f} M☉")

        print("\nSpatial distribution:")
        print(f"  Mean radius: {np.mean(r):.2f} kpc")
        print(f"  Core (<2 kpc): {np.sum(r < 2)} stars")
        print(f"  Disk (2-15 kpc): {np.sum((r >= 2) & (r < 15))} stars")
        print(f"  Periphery (>15 kpc): {np.sum(r >= 15)} stars")

        print("\nKinematics:")
        print(f"  Mean velocity: {np.mean(v):.1f} km/s")
        print(f"  Velocity dispersion: {np.std(v):.1f} km/s")

        density_core = np.sum(r < 2) / (np.pi * (2.0**2))
        density_disk = np.sum((r >= 2) & (r < 15)) / (np.pi * (15.0**2 - 2.0**2))
        print("\nDensity:")
        print(f"  Core density: {density_core:.1f} stars/kpc²")
        print(f"  Disk density: {density_disk:.1f} stars/kpc²")

    def plot_galaxy(self, mode_tag="newton"):
        if self.use_gpu:
            x = cp.asnumpy(self.x); y = cp.asnumpy(self.y); z = cp.asnumpy(self.z)
            m = cp.asnumpy(self.masses)
            vx = cp.asnumpy(self.vx); vy = cp.asnumpy(self.vy); vz = cp.asnumpy(self.vz)
        else:
            x, y, z = self.x, self.y, self.z
            m = self.masses
            vx, vy, vz = self.vx, self.vy, self.vz

        fig = plt.figure(figsize=(18, 12))

        # 1) Face-on
        ax1 = plt.subplot(2,3,1)
        colors = {'bulge':'red', 'disk':'blue', 'arms':'gold'}
        for comp in ['bulge','disk','arms']:
            mask = (self.component_labels == comp)
            ax1.scatter(x[mask], y[mask], s=1, alpha=0.5, c=colors[comp], label=comp)
        ax1.set_xlim(-self.galaxy_radius, self.galaxy_radius)
        ax1.set_ylim(-self.galaxy_radius, self.galaxy_radius)
        ax1.set_aspect('equal')
        ax1.set_title('Face-on View')
        ax1.set_xlabel('X (kpc)'); ax1.set_ylabel('Y (kpc)')
        ax1.legend()
        for radius in [2,5,10,15,20]:
            ax1.add_patch(Circle((0,0), radius, fill=False, edgecolor='gray', linestyle='--', alpha=0.3))

        # 2) Edge-on
        ax2 = plt.subplot(2,3,2)
        for comp in ['bulge','disk','arms']:
            mask = (self.component_labels == comp)
            ax2.scatter(x[mask], z[mask], s=1, alpha=0.5, c=colors[comp])
        ax2.set_xlim(-self.galaxy_radius, self.galaxy_radius)
        ax2.set_ylim(-2, 2)
        ax2.set_aspect('equal')
        ax2.set_title('Edge-on View')
        ax2.set_xlabel('X (kpc)'); ax2.set_ylabel('Z (kpc)')

        # 3) Rotation curve
        ax3 = plt.subplot(2,3,3)
        r_samp = np.linspace(0.1, self.galaxy_radius, 128)
        v_rot = self._rotation_curve_scalar(r_samp)
        ax3.plot(r_samp, v_rot, linewidth=2, label='Model (no DM)')
        r_star = np.sqrt(x*x + y*y)
        v_circ = np.sqrt(vx*vx + vy*vy)
        rbins = np.linspace(0, self.galaxy_radius, 24)
        rc, vb = [], []
        for i in range(len(rbins)-1):
            mask = (r_star >= rbins[i]) & (r_star < rbins[i+1])
            if np.count_nonzero(mask) > 15:
                rc.append(0.5*(rbins[i]+rbins[i+1]))
                vb.append(np.mean(v_circ[mask]))
        ax3.scatter(rc, vb, s=20, alpha=0.7, label='Stars')
        ax3.set_xlabel('Radius (kpc)'); ax3.set_ylabel('Velocity (km/s)')
        ax3.set_title('Rotation Curve (baseline physics)')
        ax3.grid(alpha=0.3)
        ax3.legend()

        # 4) Radial density
        ax4 = plt.subplot(2,3,4)
        hist, edges = np.histogram(r_star, bins=np.linspace(0, self.galaxy_radius, 30))
        areas = np.pi * (edges[1:]**2 - edges[:-1]**2)
        surf = hist / np.maximum(areas, 1e-12)
        ax4.semilogy(0.5*(edges[1:]+edges[:-1]), surf, linewidth=2)
        ax4.set_xlabel('Radius (kpc)'); ax4.set_ylabel('Surface Density (stars/kpc²)')
        ax4.set_title('Radial Density Profile')
        ax4.grid(alpha=0.3)

        # 5) Mass distribution
        ax5 = plt.subplot(2,3,5)
        ax5.hist(m, bins=60, edgecolor='black', alpha=0.8)
        ax5.set_yscale('log')
        ax5.set_xlabel('Stellar Mass (M☉)'); ax5.set_ylabel('Count')
        ax5.set_title('Stellar Mass Distribution')

        # 6) GPU info
        ax6 = plt.subplot(2,3,6); ax6.axis('off')
        if self.use_gpu:
            mem_free, mem_total = cp.cuda.Device().mem_info
            data_mb = (self.positions.nbytes + self.velocities.nbytes + self.masses.nbytes) / 1e6
            info = (f"MODE: {mode_tag.upper()}\n\n✅ GPU ENABLED\n"
                    f"Mem Free: {mem_free/1e9:.1f} GB\n"
                    f"Data on GPU: {data_mb:.1f} MB\n"
                    f"Stars: {self.num_stars:,}\n"
                    f"Ready for geodesic test")
            face = 'lightgreen'
        else:
            info = (f"MODE: {mode_tag.upper()}\n\n⚠️ GPU DISABLED\n"
                    f"CPU arrays in use\n"
                    f"Stars: {self.num_stars:,}")
            face = 'yellow'
        ax6.text(0.05, 0.95, info, va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor=face, alpha=0.8),
                 family='monospace', fontsize=11)

        plt.tight_layout()
        out = f"gpu_galaxy_structure_{mode_tag}.png"
        try:
            plt.savefig(out, dpi=300, bbox_inches='tight')
        except Exception:
            pass
        plt.show()

    # --------- accelerations ----------
    def _base_accel_gpu(self):
        """Return (ax, ay, az) from CBH+bulge+disk model (no DM)."""
        if self.use_gpu:
            r_cyl = cp.sqrt(self.x*self.x + self.y*self.y) + 1e-6
            r_sph = cp.sqrt(self.x*self.x + self.y*self.y + self.z*self.z) + 1e-6
            v = self._rotation_curve_gpu(r_cyl)
            a_rad = (v*v) / r_cyl  # in-plane radial accel magnitude
            ax = -a_rad * (self.x / r_cyl)
            ay = -a_rad * (self.y / r_cyl)
            # vertical: CBH restores toward midplane (approx)
            az = -self.G * self.cbh_mass * (self.z / (r_sph*r_sph*r_sph))
            return ax, ay, az
        else:
            r_cyl = np.sqrt(self.x*self.x + self.y*self.y) + 1e-6
            r_sph = np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z) + 1e-6
            v = self._rotation_curve_scalar(r_cyl)
            a_rad = (v*v) / r_cyl
            ax = -a_rad * (self.x / r_cyl)
            ay = -a_rad * (self.y / r_cyl)
            az = -self.G * self.cbh_mass * (self.z / (r_sph*r_sph*r_sph))
            return ax, ay, az

    def _geodesic_overlap_accel_gpu(self, influence_scale=0.6, coupling=1.0, tile=4096):
        """
        Geodesic "raindrop" overlap acceleration in the disk plane.
        Influence radius per star: R_i = influence_scale * (M_i)^(1/3)  [kpc]
        For neighbor j within R_i: a_i += coupling * G * |Mi - Mj| / d^2  *  unit_vector(i->j)
        (Direction toward neighbor; local Venn-slice effect.)

        Vectorized & tiled on GPU to avoid OOM. Returns (ax, ay). (No az: overlap treated as in-plane.)
        """
        if not self.use_gpu:
            # CPU fallback (slower) — use small N if CPU-only
            xp = np
        else:
            xp = cp

        N = self.num_stars
        x = self.x; y = self.y
        m = self.masses

        # Influence radius per star
        Ri = influence_scale * xp.power(m, 1.0/3.0)  # shape (N,)

        ax_total = xp.zeros(N, dtype=xp.float64)
        ay_total = xp.zeros(N, dtype=xp.float64)

        # Tiled neighbor accumulation
        # We sweep columns (neighbors) in tiles to limit memory.
        eps = 1e-5
        for j0 in range(0, N, tile):
            j1 = min(j0+tile, N)
            # tile arrays (broadcast along rows)
            xt = x[j0:j1][xp.newaxis, :]   # shape (1, T)
            yt = y[j0:j1][xp.newaxis, :]
            mt = m[j0:j1][xp.newaxis, :]

            # rows are all i (as a column)
            xi = x[:, xp.newaxis]          # (N,1)
            yi = y[:, xp.newaxis]
            mi = m[:, xp.newaxis]
            Ri_i = Ri[:, xp.newaxis]       # (N,1) broadcast vs tile cols

            dx = xt - xi                   # (N,T)
            dy = yt - yi
            d2 = dx*dx + dy*dy + eps
            d = xp.sqrt(d2)

            # within influence of i (only) — per your formulation focus on the local well
            mask = (d < Ri_i) & (d > 1e-4)

            # magnitude = coupling * G * |mi - mj| / d^2
            # Use abs mass difference for the Venn gradient strength
            mdiff = xp.abs(mi - mt)
            mag = coupling * self.G * mdiff / d2

            # unit vector toward neighbor j
            ux = dx / d
            uy = dy / d

            ax = xp.where(mask, mag * ux, 0.0)
            ay = xp.where(mask, mag * uy, 0.0)

            # sum over tile neighbors
            ax_total += ax.sum(axis=1)
            ay_total += ay.sum(axis=1)

            # free intermediates sooner (CuPy GC will handle)
            del xt, yt, mt, dx, dy, d2, d, mask, mdiff, mag, ux, uy, ax, ay

        return ax_total, ay_total

    # --------- integrator ----------
    def simulate_orbits_gpu(self,
                            mode="newton",
                            time_myr=100.0,
                            dt_myr=1.0,
                            influence_scale=0.6,
                            coupling=1.0,
                            report_tag="newton"):
        """
        Single-step Euler integrator (GPU-native when available).
        - Base accel: CBH+bulge+disk (no DM).
        - If mode == "geodesic": add neighbor-overlap accel in-plane.
        Escape condition: r > 3 * galaxy_radius
        """
        if not self.use_gpu:
            print("WARNING: GPU not available, simulation will be much slower on CPU.")

        print("\n" + "="*60)
        print("GPU-ACCELERATED ORBITAL SIMULATION")
        print("="*60)

        n_steps = int(round(time_myr / dt_myr))
        # Convert Myr to seconds to kpc/(km/s): 1 km/s = 1 kpc / 977.8 Myr
        # So dt_code in kpc per (km/s) units:
        dt_code = dt_myr / 977.8

        start = time.time()
        escaped_mask = None

        r0 = self._norm_r(self.x, self.y)

        for step in range(n_steps):
            # base accel
            ax, ay, az = self._base_accel_gpu()

            if mode == "geodesic":
                gx, gy = self._geodesic_overlap_accel_gpu(influence_scale=influence_scale,
                                                          coupling=coupling,
                                                          tile=4096 if self.use_gpu else 1024)
                ax = ax + gx
                ay = ay + gy

            # integrate (simple Euler)
            self.vx += ax * dt_code
            self.vy += ay * dt_code
            self.vz += az * dt_code

            self.x += self.vx * dt_code
            self.y += self.vy * dt_code
            self.z += self.vz * dt_code

            # progress & escaped
            if (step+1) % max(1, n_steps//10) == 0 or step == n_steps-1:
                r_now = self._norm_r(self.x, self.y)
                esc = (r_now > (3.0 * self.galaxy_radius))
                if self.use_gpu:
                    esc_count = int(cp.sum(esc).get())  # type: ignore
                else:
                    esc_count = int(np.sum(esc))
                progress = 100.0 * (step+1) / n_steps
                print(f"\r  Progress: {progress:5.1f}%, Escaped: {esc_count}", end="", flush=True)
                escaped_mask = esc  # keep latest

        print()
        elapsed = time.time() - start
        print(f"\nSimulation complete in {elapsed:.2f} seconds")
        if elapsed > 1e-6:
            print(f"Speed: {self.num_stars * n_steps / max(elapsed,1e-6):.0f} star-steps/sec")
        if self.use_gpu:
            data_mb = (self.positions.nbytes + self.velocities.nbytes + self.masses.nbytes) / 1e6
            print(f"GPU Memory used: {data_mb:.1f} MB")

        # return escaped boolean (CPU ndarray)
        if escaped_mask is None:
            if self.use_gpu:
                escaped_mask = cp.zeros(self.num_stars, dtype=cp.bool_)
            else:
                escaped_mask = np.zeros(self.num_stars, dtype=bool)

        if self.use_gpu:
            return cp.asnumpy(escaped_mask), cp.asnumpy(self.initial_radius)
        else:
            return escaped_mask, self.initial_radius

# -----------------------------
# Utility: save escape CSV + print region stats
# -----------------------------
def save_escape_report(filename, escaped, r_init, x, y, vx, vy, masses, galaxy_radius):
    import csv
    N = len(escaped)
    regions = np.empty(N, dtype='<U10')
    regions[r_init < 2.0] = 'core'
    regions[(r_init >= 2.0) & (r_init < 15.0)] = 'disk'
    regions[r_init >= 15.0] = 'periphery'

    # Summary counts
    total_esc = int(np.sum(escaped))
    core_esc  = int(np.sum(escaped & (regions == 'core')))
    disk_esc  = int(np.sum(escaped & (regions == 'disk')))
    peri_esc  = int(np.sum(escaped & (regions == 'periphery')))
    total = N
    print("\n" + "-"*60)
    print("ESCAPE SUMMARY")
    print("-"*60)
    print(f"Total escaped: {total_esc}/{total} ({100*total_esc/total:.1f}%)")
    for name, cnt, denom in [
        ('Core   (<2 kpc)', core_esc, int(np.sum(regions=='core'))),
        ('Disk   (2–15 kpc)', disk_esc, int(np.sum(regions=='disk'))),
        ('Periph (>15 kpc)', peri_esc, int(np.sum(regions=='periphery'))),
    ]:
        pct = (100*cnt/denom) if denom>0 else 0.0
        print(f"{name:16s}: {cnt}/{denom} ({pct:.1f}%)")

    # Write CSV
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','x','y','vx','vy','mass','r_init_kpc','region','escaped'])
        for i in range(N):
            w.writerow([i, x[i], y[i], vx[i], vy[i], masses[i], r_init[i], regions[i], int(escaped[i])])
    print(f"\nSaved: {filename}")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Newton vs Geodesic GPU Galaxy Test")
    ap.add_argument('--mode', choices=['newton','geodesic'], default='newton')
    ap.add_argument('--num-stars', type=int, default=10000)
    ap.add_argument('--time-myr', type=float, default=100.0)
    ap.add_argument('--dt-myr', type=float, default=1.0)
    # geodesic params (leave defaults; tuned to your description)
    ap.add_argument('--influence-scale', type=float, default=0.6)  # kpc * (M/Msun)^(1/3)
    ap.add_argument('--coupling', type=float, default=1.0)
    args = ap.parse_args()

    if GPU_AVAILABLE:
        print("🚀 GPU ACCELERATION AVAILABLE!")
    else:
        print("⚠️ WARNING: GPU not available — continuing on CPU.")

    # Build galaxy (same initial conditions for both modes if you run separately)
    galaxy = RealisticGalaxy(
        num_stars=args.num_stars if GPU_AVAILABLE else min(args.num_stars, 3000),
        cbh_mass=4.15e6,
        bulge_mass=2e10,
        disk_mass=6e10,
        galaxy_radius=20.0,
        scale_height=0.3,
        num_arms=4,
        arm_pitch_deg=12.0,
        bar_strength=0.3,
        use_gpu=True if GPU_AVAILABLE else False,
        rng_seed=42
    )

    # Plot diagnostic
    galaxy.plot_galaxy(mode_tag=args.mode)

    # Run integrator
    escaped, r_init = galaxy.simulate_orbits_gpu(
        mode=args.mode,
        time_myr=args.time_myr,
        dt_myr=args.dt_myr,
        influence_scale=args.influence_scale,
        coupling=args.coupling,
        report_tag=args.mode
    )

    # Pull CPU data for report
    if galaxy.use_gpu:
        x = cp.asnumpy(galaxy.x); y = cp.asnumpy(galaxy.y)
        vx = cp.asnumpy(galaxy.vx); vy = cp.asnumpy(galaxy.vy)
        masses = cp.asnumpy(galaxy.masses)
    else:
        x, y = galaxy.x, galaxy.y
        vx, vy = galaxy.vx, galaxy.vy
        masses = galaxy.masses

    # Save report
    save_escape_report(f"escapes_{args.mode}.csv", escaped, r_init, x, y, vx, vy, masses, galaxy.galaxy_radius)

    print("\n" + "="*60)
    print("GALAXY MODEL COMPLETE")
    print("="*60)
    print("Ready for direct Newton vs Geodesic comparison. Run both modes and compare escape stats.")

if __name__ == "__main__":
    main()
