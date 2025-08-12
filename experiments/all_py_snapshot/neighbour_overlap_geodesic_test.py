# neighbour_overlap_geodesic_test.py
# One file: core class + runners + density-normalized neighbor metric.
# GPU with CuPy if available; falls back to NumPy seamlessly.

import math
import sys
import time
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe for headless / scripts
import matplotlib.pyplot as plt

# Try GPU
try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
    xp = cp
except Exception:
    GPU_AVAILABLE = False
    cp = None  # type: ignore
    xp = np

# High-contrast visuals
plt.style.use("dark_background")

__all__ = [
    "NeighborOverlapTest",
    "run_neighbor_overlap_test",
    "compare_scaling_laws",
    "run_enhanced_neighbor_test",
    "test_coupling_strength_sweep",
    "run_multi_seed_radius_sweep",
]

def _asnumpy(arr):
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def _rand_uniform(low, high, size, module):
    return module.random.uniform(low, high, size)

def _pairwise_sqdist(P):
    # P: (N,3)
    diff = P[:, xp.newaxis, :] - P[xp.newaxis, :, :]
    return xp.sum(diff * diff, axis=2)

def _uniform_points_in_sphere(n, radius, module):
    # Rejection-sample in batches to avoid excess host<->device syncs
    pts = []
    batch = max(256, int(n * 2))
    r2max = radius * radius
    while sum(p.shape[0] for p in pts) < n:
        xyz = _rand_uniform(-radius, radius, (batch, 3), module)
        r2 = module.sum(xyz * xyz, axis=1)
        kept = xyz[r2 <= r2max]
        pts.append(kept)
        # modest ramp for small n
        if len(pts) == 1:
            batch = int(batch * 1.2)
    out = module.concatenate(pts, axis=0)[:n]
    return out

def _mean_spacing(R: float, N: int) -> float:
    vol = (4.0 / 3.0) * math.pi * (R ** 3)
    return (vol / float(N)) ** (1.0 / 3.0)

class NeighborOverlapTest:
    """
    Density-normalized neighbor overlap metric.
      - Generate N points ~ uniform in sphere radius R
      - Pairwise distances -> count neighbors within d_nbr
      - avg_overlap = mean(count_i)
      - retention = 1 / (1 + avg_overlap)

    GPU acceleration automatic with CuPy.
    """

    def __init__(self, num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6, seed=42):
        self.num_stars = int(num_stars)
        self.galaxy_radius = float(galaxy_radius)
        self.cbh_mass = float(cbh_mass)
        self.seed = int(seed)
        # Seed both RNGs for reproducibility
        np.random.seed(self.seed)
        if GPU_AVAILABLE:
            cp.random.seed(self.seed)

    def generate_positions(self, radius: float):
        return _uniform_points_in_sphere(self.num_stars, float(radius), xp)

    def compute_neighbor_overlap(self, positions, neighbor_distance: float) -> Tuple[float, float]:
        d2 = _pairwise_sqdist(positions)
        N = positions.shape[0]
        # Avoid NaNs: set self-distances to +inf directly
        xp.fill_diagonal(d2, xp.inf)
        within = d2 <= (neighbor_distance * neighbor_distance)
        counts = xp.sum(within, axis=1)
        avg_overlap = xp.mean(counts)
        avg_overlap = float(_asnumpy(avg_overlap))
        retention_fraction = 1.0 / (1.0 + avg_overlap)
        return avg_overlap, retention_fraction

    def run_radius_sweep(
        self,
        scaling_law: str = "sqrt",
        neighbor_eta: float = 1.0,
        r_min_factor: float = 0.1,
        r_max_factor: float = 2.0,
        samples: int = 10,
    ) -> Tuple[float, List[Tuple[float, float, float, float]]]:
        """
        Use density-normalized neighbor distance: d_nbr = eta * mean_spacing(R).
        Returns:
          baseline_retention,
          results = [(R, avg_overlap, retention, improvement), ...]
        """
        samples = int(samples)
        baseline_radius = self.galaxy_radius / 2.0

        def d_nbr(R: float) -> float:
            return neighbor_eta * _mean_spacing(R, self.num_stars)

        # Baseline
        d_base = d_nbr(baseline_radius)
        base_pos = self.generate_positions(baseline_radius)
        _, baseline_retention = self.compute_neighbor_overlap(base_pos, d_base)

        radii = np.linspace(self.galaxy_radius * r_min_factor,
                            self.galaxy_radius * r_max_factor,
                            samples)

        results: List[Tuple[float, float, float, float]] = []
        for R in radii:
            dn = d_nbr(float(R))
            pos = self.generate_positions(float(R))
            avg_overlap, retention = self.compute_neighbor_overlap(pos, dn)

            # Map retention -> improvement
            num = max(retention, 1e-12)
            den = max(baseline_retention, 1e-12)
            if scaling_law == "sqrt":
                improvement = math.sqrt(num / den)
            elif scaling_law == "log":
                improvement = math.log(num / den + 1.0)
            else:  # 'linear' or unknown
                improvement = num / den

            results.append((float(R), float(avg_overlap), float(retention), float(improvement)))

        return float(baseline_retention), results

    def plot_radius_analysis(
        self,
        baseline_retention: float,
        results: List[Tuple[float, float, float, float]],
        outfile: str = "neighbor_overlap_analysis.png",
        title_suffix: str = "",
    ) -> Tuple[float, float, float]:
        radii = [r for r, _, _, _ in results]
        improvements = [imp for _, _, _, imp in results]
        retentions = [ret for _, _, ret, _ in results]

        best_idx = int(np.argmax(improvements))
        best_radius = radii[best_idx]
        best_retention = retentions[best_idx]
        best_improvement = improvements[best_idx]

        plt.figure(figsize=(12, 7))
        plt.plot(radii, improvements, marker='o', linewidth=2.5, label='Improvement')
        plt.axhline(1.0, linestyle='--', linewidth=1.5, label='Baseline (1.0)')
        plt.axvline(best_radius, linestyle='--', linewidth=1.5, label=f'Best Radius = {best_radius:.1f}')
        plt.xlabel('Galaxy Radius (units)')
        plt.ylabel('Improvement Factor')
        ttl = 'Radius Sweep — Neighbor Overlap Improvement'
        if title_suffix:
            ttl += f' ({title_suffix})'
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outfile, dpi=170)
        plt.close()

        return float(best_radius), float(best_retention), float(best_improvement)


# ---------- Convenience runners ----------

def run_neighbor_overlap_test(scaling_law='sqrt', neighbor_eta=1.0):
    test = NeighborOverlapTest(num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6)
    base, res = test.run_radius_sweep(scaling_law=scaling_law, neighbor_eta=neighbor_eta)
    br, brt, imp = test.plot_radius_analysis(base, res, outfile='neighbor_overlap_analysis.png',
                                             title_suffix=f'η={neighbor_eta}')
    return br, brt, imp

def compare_scaling_laws(neighbor_eta=1.0) -> Dict[str, Tuple[float, float, float]]:
    laws = ['sqrt', 'log', 'linear']
    out: Dict[str, Tuple[float, float, float]] = {}
    for law in laws:
        br, brt, imp = run_neighbor_overlap_test(law, neighbor_eta=neighbor_eta)
        out[law] = (br, brt, imp)
    return out

def run_enhanced_neighbor_test(scaling_law='sqrt', neighbor_eta=1.0):
    print("\n" + "=" * 60)
    print("🚀 ENHANCED NEIGHBOR OVERLAP TEST")
    print("Density-normalized neighbors (d_nbr = η·mean_spacing).")
    print("=" * 60)

    test = NeighborOverlapTest(num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6)
    base, res = test.run_radius_sweep(scaling_law=scaling_law, neighbor_eta=neighbor_eta)
    best_R, best_ret, best_imp = test.plot_radius_analysis(
        base, res,
        outfile='neighbor_overlap_analysis.png',
        title_suffix=f'η={neighbor_eta}'
    )
    print(f"Best radius: {best_R:.1f}")
    print(f"Best retention: {best_ret:.6f}")
    print(f"Best improvement: {best_imp:.6f}")
    if GPU_AVAILABLE:
        print("GPU: CuPy detected ✅")
    else:
        print("GPU: not available, using CPU")
    return best_imp

def test_coupling_strength_sweep(etas: np.ndarray = None, scaling_law='sqrt'):
    """
    Sweep η (dimensionless coupling controlling neighbor radius).
    """
    if etas is None:
        etas = np.linspace(0.5, 2.0, 10)

    improvements = []
    for eta in etas:
        test = NeighborOverlapTest(num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6)
        base, res = test.run_radius_sweep(scaling_law=scaling_law, neighbor_eta=float(eta))
        _, _, imp = test.plot_radius_analysis(
            base, res, outfile=f'neighbor_overlap_analysis_eta_{eta:.2f}.png',
            title_suffix=f'η={eta:.2f}'
        )
        improvements.append(imp)

    best_idx = int(np.argmax(improvements))
    optimal_eta = float(etas[best_idx])
    max_improvement = float(improvements[best_idx])

    print(f"Optimal η: {optimal_eta:.2f}")
    print(f"Max Improvement: {max_improvement:.6f}")
    return optimal_eta, max_improvement

def run_multi_seed_radius_sweep(
    seeds: List[int],
    num_stars=400,
    galaxy_radius=15000.0,
    neighbor_eta=1.0,
    scaling_law='sqrt',
    r_min_factor=0.1,
    r_max_factor=2.0,
    samples=15,
    outfile='neighbor_overlap_analysis_multiseed.png'
):
    """
    Average improvement curves over multiple seeds for stability.
    Saves a single averaged plot with ±1σ shading.
    """
    all_radii = None
    all_imps = []

    for s in seeds:
        test = NeighborOverlapTest(num_stars=num_stars, galaxy_radius=galaxy_radius, seed=int(s))
        base, res = test.run_radius_sweep(
            scaling_law=scaling_law,
            neighbor_eta=neighbor_eta,
            r_min_factor=r_min_factor,
            r_max_factor=r_max_factor,
            samples=samples,
        )
        radii = np.array([r for r, _, _, _ in res], dtype=float)
        imps = np.array([imp for _, _, _, imp in res], dtype=float)
        if all_radii is None:
            all_radii = radii
        all_imps.append(imps)

    all_imps = np.vstack(all_imps)  # (nseeds, nsamples)
    mean_imp = all_imps.mean(axis=0)
    std_imp = all_imps.std(axis=0)

    # Plot averaged curve
    plt.figure(figsize=(12, 7))
    plt.plot(all_radii, mean_imp, marker='o', linewidth=2.5, label='Mean Improvement (multi-seed)')
    plt.fill_between(all_radii, mean_imp - std_imp, mean_imp + std_imp, alpha=0.2, label='±1σ')
    plt.axhline(1.0, linestyle='--', linewidth=1.5, label='Baseline (1.0)')
    best_idx = int(np.argmax(mean_imp))
    best_R = float(all_radii[best_idx])
    plt.axvline(best_R, linestyle='--', linewidth=1.5, label=f'Best Radius ≈ {best_R:.1f}')
    plt.xlabel('Galaxy Radius (units)')
    plt.ylabel('Improvement Factor')
    plt.title(f'Radius Sweep — Mean Improvement (η={neighbor_eta}, seeds={len(seeds)})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=170)
    plt.close()

    print(f"[Multi-seed] Best radius (mean curve): {best_R:.1f}")
    return best_R, float(mean_imp[best_idx])


# ---------- Script entry ----------

if __name__ == "__main__":
    print("Starting enhanced neighbor overlap analysis...")

    # Main demo: density-normalized sweep with η=1.0
    best_improvement = run_enhanced_neighbor_test(scaling_law='sqrt', neighbor_eta=1.0)
# neighbour_overlap_geodesic_test.py
# Density-normalized neighbor overlap with GPU (CuPy) accel.
# Uses a single unit-sphere point cloud per seed and scales by R to kill redraw noise.
# Includes analytic expectation for sanity checks.

import math
from typing import List, Tuple, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try GPU
try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
    xp = cp
except Exception:
    GPU_AVAILABLE = False
    cp = None  # type: ignore
    xp = np

plt.style.use("dark_background")

__all__ = [
    "NeighborOverlapTest",
    "run_neighbor_overlap_test",
    "compare_scaling_laws",
    "run_enhanced_neighbor_test",
    "test_coupling_strength_sweep",
    "run_multi_seed_radius_sweep",
]

def _asnumpy(arr):
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

def _pairwise_sqdist(P):
    # P: (N,3)
    diff = P[:, xp.newaxis, :] - P[xp.newaxis, :, :]
    return xp.sum(diff * diff, axis=2)

def _uniform_points_in_unit_sphere(n: int, module) -> "np.ndarray | cp.ndarray":
    # Rejection sample in the unit sphere
    pts = []
    batch = max(256, int(n * 2))
    while sum(p.shape[0] for p in pts) < n:
        xyz = module.random.uniform(-1.0, 1.0, (batch, 3))
        r2 = module.sum(xyz * xyz, axis=1)
        kept = xyz[r2 <= 1.0]
        pts.append(kept)
        if len(pts) == 1:
            batch = int(batch * 1.2)
    out = module.concatenate(pts, axis=0)[:n]
    return out

def _mean_spacing(R: float, N: int) -> float:
    vol = (4.0 / 3.0) * math.pi * (R ** 3)
    return (vol / float(N)) ** (1.0 / 3.0)

def analytic_expected_retention(eta: float) -> float:
    # E[count] = (4/3)π η^3; retention ≈ 1 / (1 + E[count])
    m = (4.0 / 3.0) * math.pi * (eta ** 3)
    return 1.0 / (1.0 + m)

class NeighborOverlapTest:
    """
    Density-normalized neighbor overlap metric.
    One unit-sphere point cloud per seed; scaled by R for all radii.
    """

    def __init__(self, num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6, seed=42):
        self.num_stars = int(num_stars)
        self.galaxy_radius = float(galaxy_radius)
        self.cbh_mass = float(cbh_mass)
        self.seed = int(seed)
        # Seed both RNGs
        np.random.seed(self.seed)
        if GPU_AVAILABLE:
            cp.random.seed(self.seed)
        # Generate canonical unit-sphere sample once
        self.unit_points = _uniform_points_in_unit_sphere(self.num_stars, xp)

    def positions_at_radius(self, R: float):
        # Scale unit-sphere sample to radius R
        return self.unit_points * float(R)

    def compute_neighbor_overlap(self, positions, neighbor_distance: float) -> Tuple[float, float]:
        d2 = _pairwise_sqdist(positions)
        N = positions.shape[0]
        xp.fill_diagonal(d2, xp.inf)  # avoid NaNs
        within = d2 <= (neighbor_distance * neighbor_distance)
        counts = xp.sum(within, axis=1)
        avg_overlap = xp.mean(counts)
        avg_overlap = float(_asnumpy(avg_overlap))
        retention_fraction = 1.0 / (1.0 + avg_overlap)
        return avg_overlap, retention_fraction

    def run_radius_sweep(
        self,
        scaling_law: str = "sqrt",
        neighbor_eta: float = 1.0,
        r_min_factor: float = 0.1,
        r_max_factor: float = 2.0,
        samples: int = 10,
    ) -> Tuple[float, List[Tuple[float, float, float, float]]]:
        """
        d_nbr = eta * mean_spacing(R). Reuses the *same* cloud, scaled by R.
        Returns:
          baseline_retention,
          [(R, avg_overlap, retention, improvement), ...]
        """
        samples = int(samples)
        baseline_radius = self.galaxy_radius / 2.0

        def d_nbr(R: float) -> float:
            return neighbor_eta * _mean_spacing(R, self.num_stars)

        # Baseline at R0
        R0 = baseline_radius
        d0 = d_nbr(R0)
        pos0 = self.positions_at_radius(R0)
        _, baseline_ret = self.compute_neighbor_overlap(pos0, d0)

        radii = np.linspace(self.galaxy_radius * r_min_factor,
                            self.galaxy_radius * r_max_factor,
                            samples)

        results: List[Tuple[float, float, float, float]] = []
        for R in radii:
            dn = d_nbr(float(R))
            pos = self.positions_at_radius(float(R))
            avg_overlap, retention = self.compute_neighbor_overlap(pos, dn)

            num = max(retention, 1e-12)
            den = max(baseline_ret, 1e-12)
            if scaling_law == "sqrt":
                improvement = math.sqrt(num / den)
            elif scaling_law == "log":
                improvement = math.log(num / den + 1.0)
            else:
                improvement = num / den

            results.append((float(R), float(avg_overlap), float(retention), float(improvement)))

        return float(baseline_ret), results

    def plot_radius_analysis(
        self,
        baseline_retention: float,
        results: List[Tuple[float, float, float, float]],
        outfile: str = "neighbor_overlap_analysis.png",
        title_suffix: str = "",
    ) -> Tuple[float, float, float]:
        radii = [r for r, _, _, _ in results]
        improvements = [imp for _, _, _, imp in results]
        retentions = [ret for _, _, ret, _ in results]

        best_idx = int(np.argmax(improvements))
        best_radius = radii[best_idx]
        best_retention = retentions[best_idx]
        best_improvement = improvements[best_idx]

        plt.figure(figsize=(12, 7))
        plt.plot(radii, improvements, marker='o', linewidth=2.5, label='Improvement')
        plt.axhline(1.0, linestyle='--', linewidth=1.5, label='Baseline (1.0)')
        plt.axvline(best_radius, linestyle='--', linewidth=1.5, label=f'Best Radius = {best_radius:.1f}')
        plt.xlabel('Galaxy Radius (units)')
        plt.ylabel('Improvement Factor')
        ttl = 'Radius Sweep — Neighbor Overlap Improvement'
        if title_suffix:
            ttl += f' ({title_suffix})'
        plt.title(ttl)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outfile, dpi=170)
        plt.close()

        return float(best_radius), float(best_retention), float(best_improvement)


# ---------- Convenience runners ----------

def run_neighbor_overlap_test(scaling_law='sqrt', neighbor_eta=1.0):
    test = NeighborOverlapTest(num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6)
    base, res = test.run_radius_sweep(scaling_law=scaling_law, neighbor_eta=neighbor_eta)
    br, brt, imp = test.plot_radius_analysis(base, res, outfile='neighbor_overlap_analysis.png',
                                             title_suffix=f'η={neighbor_eta:.2f}')
    return br, brt, imp

def compare_scaling_laws(neighbor_eta=1.0) -> Dict[str, Tuple[float, float, float]]:
    laws = ['sqrt', 'log', 'linear']
    out: Dict[str, Tuple[float, float, float]] = {}
    for law in laws:
        br, brt, imp = run_neighbor_overlap_test(law, neighbor_eta=neighbor_eta)
        out[law] = (br, brt, imp)
    return out

def run_enhanced_neighbor_test(scaling_law='sqrt', neighbor_eta=1.0):
    print("\n" + "=" * 60)
    print("🚀 ENHANCED NEIGHBOR OVERLAP TEST")
    print("Density-normalized neighbors (d_nbr = η·mean_spacing).")
    print("Reusing one unit-sphere cloud scaled by R to reduce noise.")
    print("=" * 60)

    test = NeighborOverlapTest(num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6)
    base, res = test.run_radius_sweep(scaling_law=scaling_law, neighbor_eta=neighbor_eta)
    best_R, best_ret, best_imp = test.plot_radius_analysis(
        base, res, outfile='neighbor_overlap_analysis.png', title_suffix=f'η={neighbor_eta:.2f}'
    )

    exp_ret = analytic_expected_retention(neighbor_eta)
    print(f"Expected retention (analytic): {exp_ret:.6f}")
    print(f"Baseline retention (empirical): {base:.6f}")
    print(f"Best radius: {best_R:.1f}")
    print(f"Best retention: {best_ret:.6f}")
    print(f"Best improvement: {best_imp:.6f}")
    if GPU_AVAILABLE:
        print("GPU: CuPy detected ✅")
    else:
        print("GPU: not available, using CPU")
    return best_imp

def test_coupling_strength_sweep(etas: np.ndarray = None, scaling_law='sqrt'):
    if etas is None:
        etas = np.linspace(0.5, 2.0, 10)

    improvements = []
    for eta in etas:
        test = NeighborOverlapTest(num_stars=400, galaxy_radius=15000.0, cbh_mass=4.15e6)
        base, res = test.run_radius_sweep(scaling_law=scaling_law, neighbor_eta=float(eta))
        _, _, imp = test.plot_radius_analysis(
            base, res, outfile=f'neighbor_overlap_analysis_eta_{eta:.2f}.png',
            title_suffix=f'η={eta:.2f}'
        )
        improvements.append(imp)

    best_idx = int(np.argmax(improvements))
    optimal_eta = float(etas[best_idx])
    max_improvement = float(improvements[best_idx])

    print(f"Optimal η: {optimal_eta:.2f}")
    print(f"Max Improvement: {max_improvement:.6f}")
    return optimal_eta, max_improvement

def run_multi_seed_radius_sweep(
    seeds: List[int],
    num_stars=400,
    galaxy_radius=15000.0,
    neighbor_eta=1.0,
    scaling_law='sqrt',
    r_min_factor=0.1,
    r_max_factor=2.0,
    samples=25,
    outfile='neighbor_overlap_analysis_multiseed.png'
):
    """
    Average improvement curves over multiple seeds with the 'fixed cloud' method.
    """
    all_radii = None
    all_imps = []

    for s in seeds:
        test = NeighborOverlapTest(num_stars=num_stars, galaxy_radius=galaxy_radius, seed=int(s))
        base, res = test.run_radius_sweep(
            scaling_law=scaling_law,
            neighbor_eta=neighbor_eta,
            r_min_factor=r_min_factor,
            r_max_factor=r_max_factor,
            samples=samples,
        )
        radii = np.array([r for r, _, _, _ in res], dtype=float)
        imps = np.array([imp for _, _, _, imp in res], dtype=float)
        if all_radii is None:
            all_radii = radii
        all_imps.append(imps)

    all_imps = np.vstack(all_imps)  # (nseeds, nsamples)
    mean_imp = all_imps.mean(axis=0)
    std_imp = all_imps.std(axis=0)

    # Plot averaged curve
    plt.figure(figsize=(12, 7))
    plt.plot(all_radii, mean_imp, marker='o', linewidth=2.5, label='Mean Improvement (multi-seed)')
    plt.fill_between(all_radii, mean_imp - std_imp, mean_imp + std_imp, alpha=0.2, label='±1σ')
    plt.axhline(1.0, linestyle='--', linewidth=1.5, label='Baseline (1.0)')
    best_idx = int(np.argmax(mean_imp))
    best_R = float(all_radii[best_idx])
    plt.axvline(best_R, linestyle='--', linewidth=1.5, label=f'Best Radius ≈ {best_R:.1f}')
    plt.xlabel('Galaxy Radius (units)')
    plt.ylabel('Improvement Factor')
    plt.title(f'Radius Sweep — Mean Improvement (η={neighbor_eta}, seeds={len(seeds)})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=170)
    plt.close()

    print(f"[Multi-seed] Best radius (mean curve): {best_R:.1f}")
    return best_R, float(mean_imp[best_idx])


# ---------- Script entry ----------

if __name__ == "__main__":
    print("Starting enhanced neighbor overlap analysis...")

    # Main demo with fixed cloud and analytic check
    best_improvement = run_enhanced_neighbor_test(scaling_law='sqrt', neighbor_eta=1.0)

    # Sweep η (dimensionless coupling)
    optimal_eta, max_improvement = test_coupling_strength_sweep()

    # Optional: uncomment to average over multiple seeds (smoother curve)
    # seeds = [41, 42, 43, 44, 45]
    # run_multi_seed_radius_sweep(seeds, num_stars=1000, samples=25, neighbor_eta=1.0)

    print("\nDone.")

    # Sweep η (dimensionless coupling)
    optimal_eta, max_improvement = test_coupling_strength_sweep()

    # Optional multi-seed average for stability (uncomment to use)
    # seeds = [41, 42, 43, 44, 45]
    # run_multi_seed_radius_sweep(seeds, num_stars=1000, samples=25, neighbor_eta=1.0)

    print("\nDone.")
