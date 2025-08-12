import numpy as np

# Key cosmic times [s]
t_BBN_start = 1.0         # ~ 1 s
t_BBN_end   = 200.0       # ~ 200 s
t_eq        = 1.6e12      # matter-radiation equality ~ 50,000 yr
t_rec       = 1.2e13      # recombination ~ 380,000 yr

# Scale factor normalization (arbitrary; ratios matter)
a_BBN_ref = 1.0

def a_of_t(t):
    """
    Piecewise scale factor a(t):
    - Radiation era: a ∝ t^(1/2) up to equality t_eq
    - Matter era:    a ∝ t^(2/3) after t_eq
    Continuity at t_eq is enforced for a(t) (not its derivative).
    """
    a_eq = a_BBN_ref * (t_eq / t_BBN_start)**0.5
    if np.isscalar(t):
        if t <= t_eq:
            return a_BBN_ref * (t / t_BBN_start)**0.5
        else:
            return a_eq * (t / t_eq)**(2.0/3.0)
    else:
        t = np.array(t, dtype=float)
        a = np.empty_like(t, dtype=float)
        msk = (t <= t_eq)
        a[msk] = a_BBN_ref * (t[msk] / t_BBN_start)**0.5
        a[~msk] = a_eq * (t[~msk] / t_eq)**(2.0/3.0)
        return a

def N_inside(a, eps0, n, a0):
    """
    Lapse for the 'inside' worldline:
    N(0,t) = [1 + eps(t)]^-1/2, with eps(t) = eps0 * (a/a0)^-n
    """
    eps = eps0 * (a / a0)**(-n)
    return 1.0 / np.sqrt(1.0 + eps)

def N_outside():
    """Outside worldline has N=1 (no TDF)."""
    return 1.0

def integrate_delta_tau(eps0=0.05, n=6.0, t0=1.0, tmin=1.0, tmax=t_rec, steps=200000):
    """
    Integrate the differential proper time Δτ between 'inside' (slow clock) and
    'outside' (normal clock) from tmin to tmax. Also compute constraint metrics.

    Parameters
    ----------
    eps0 : float
        Initial amplitude of TDF at reference time t0.
    n : float
        Power-law decay index for epsilon(t) ∝ a(t)^(-n).
    t0 : float
        Reference time [s] for defining a0 = a(t0).
    tmin, tmax : float
        Integration limits in seconds.
    steps : int
        Number of log-spaced steps in t for the integral.

    Returns
    -------
    delta_tau : float
        Total Δτ in seconds (negative means inside lags).
    avg_dev_BBN : float
        Average |N-1| during BBN window [1, 200] s.
    dev_rec : float
        |N-1| at recombination.
    """
    t = np.logspace(np.log10(tmin), np.log10(tmax), steps)
    a0 = a_of_t(t0)
    a = a_of_t(t)
    N_in = N_inside(a, eps0, n, a0)
    N_out = N_outside()
    dN = N_in - N_out  # typically ≤ 0

    # Trapezoidal integral in t
    dt = np.diff(t)
    mid = 0.5 * (dN[:-1] + dN[1:])
    delta_tau = np.sum(mid * dt)

    # Constraints checks
    # 1) Average |N-1| during BBN
    t_BBN = np.logspace(np.log10(t_BBN_start), np.log10(t_BBN_end), 5000)
    a_BBN = a_of_t(t_BBN)
    N_BBN = N_inside(a_BBN, eps0, n, a0)
    avg_dev_BBN = np.trapz(np.abs(N_BBN - 1.0), t_BBN) / (t_BBN[-1] - t_BBN[0])

    # 2) By recombination, |N-1|
    a_rec = a_of_t(t_rec)
    N_rec = N_inside(a_rec, eps0, n, a0)
    dev_rec = abs(N_rec - 1.0)

    return delta_tau, avg_dev_BBN, dev_rec

def fmt_time(seconds):
    """Format seconds as seconds, days, and years for intuition."""
    s = float(seconds)
    days = s / 86400.0
    years = s / (365.25 * 86400.0)
    return f"{s:.3e} s ({days:.3e} days, {years:.3e} yr)"

if __name__ == "__main__":
    print("TDF toy model: differential proper time between inside and outside")
    print("Columns: eps0, n -> |Δτ| [seconds, days, years], <|N-1|>_BBN, |N-1|_rec\n")

    # Sweep a wider grid of eps0 and n
    eps0_list = [0.01, 0.05, 0.10, 0.20]
    n_list = [4.0, 6.0, 8.0, 10.0]

    for eps0 in eps0_list:
        for n in n_list:
            d_tau, avgBBN, devRec = integrate_delta_tau(eps0=eps0, n=n)
            # Report absolute magnitude of Δτ for readability
            print(f"eps0={eps0:>5.3f}, n={n:>4.1f} -> |Δτ| = {fmt_time(abs(d_tau))}, "
                  f"<|N-1|>_BBN={avgBBN:.3e}, |N-1|_rec={devRec:.3e}")

    print("\nInterpretation:")
    print("- |Δτ| is the total extra proper time the 'outside' accumulates over the 'inside' from 1 s to recombination.")
    print("- <|N-1|>_BBN should be ≲ few×10^-2 to satisfy BBN; values printed here are much smaller (safe).")
    print("- |N-1|_rec should be ≲ 1e-5 to satisfy CMB isotropy; values here are effectively zero under these parameters.")