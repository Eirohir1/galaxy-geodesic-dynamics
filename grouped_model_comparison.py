import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# === Load galaxy classification ===
classification = pd.read_csv("classified_galaxies.csv")
classification_dict = dict(zip(
    classification["Filename"].str.replace("_rotmod.dat", "", regex=False),
    classification["GalaxyType"]
))

# === Kernel generation ===
def geodesic_kernel(r, cutoff):
    k = np.exp(-r / cutoff)
    return k / trapezoid(k, r)

# === Model Functions ===
def model_geodesic(r, M, cutoff):
    # Simulates geodesic convolution with decay scale "cutoff"
    norm_kernel = geodesic_kernel(r, cutoff)
    v = np.sqrt(M * np.convolve(norm_kernel, np.ones_like(r), mode='same'))
    return v

def model_mond(r, M):
    a0 = 1.2e-10
    return np.sqrt((M * 1e-10) / (r + 1e-5) * (1 / (1 + (a0 * r / (M * 1e-10)))))

# === Metrics ===
def compute_metrics(y_true, y_pred, num_params):
    residual = y_true - y_pred
    chi2 = np.sum((residual)**2 / (y_true + 1e-5))
    dof = len(y_true) - num_params
    aic = chi2 + 2 * num_params
    bic = chi2 + num_params * np.log(len(y_true))
    return chi2 / dof, aic, bic

# === Determine kernel cutoff based on type ===
def get_cutoff(gtype):
    if gtype == "spiral":
        return 0.3
    elif gtype == "super-spiral":
        return 0.6
    elif gtype == "dwarf":
        return 0.15
    else:
        return 0.3

# === Main loop ===
results = []

for file in os.listdir():
    if not file.endswith('_rotmod.dat'):
        continue
    galaxy = file.replace('_rotmod.dat', '')
    gtype = classification_dict.get(galaxy, 'uncertain')

    try:
        data = np.loadtxt(file)
        r = data[:, 0]
        v_obs = data[:, 1]

        # === Fit GEODESIC model ===
        cutoff = get_cutoff(gtype)
        popt_geo, _ = curve_fit(lambda r, M: model_geodesic(r, M, cutoff), r, v_obs, p0=[1])
        v_geo = model_geodesic(r, popt_geo[0], cutoff)
        chi2_geo, aic_geo, bic_geo = compute_metrics(v_obs, v_geo, 1)

        # === Fit MOND model ===
        popt_mond, _ = curve_fit(model_mond, r, v_obs, p0=[1])
        v_mond = model_mond(r, popt_mond[0])
        chi2_mond, aic_mond, bic_mond = compute_metrics(v_obs, v_mond, 1)

        results.append({
            'Galaxy': galaxy,
            'Type': gtype,
            'GEODESIC_chi2': chi2_geo,
            'GEODESIC_AIC': aic_geo,
            'GEODESIC_BIC': bic_geo,
            'MOND_chi2': chi2_mond,
            'MOND_AIC': aic_mond,
            'MOND_BIC': bic_mond
        })

    except Exception as e:
        print(f"❌ Failed to process {galaxy}: {e}")

# === Save and group results ===
df = pd.DataFrame(results)
df.to_csv("grouped_model_comparison_results.csv", index=False)

grouped = df.groupby("Type").agg({
    'GEODESIC_chi2': ['mean', 'median'],
    'MOND_chi2': ['mean', 'median'],
    'GEODESIC_AIC': 'mean',
    'MOND_AIC': 'mean',
    'GEODESIC_BIC': 'mean',
    'MOND_BIC': 'mean'
}).round(2)

print("\n📊 Grouped Summary by Galaxy Type:")
print(grouped)
