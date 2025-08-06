import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

# === Load galaxy classification ===
classification_file = 'classified_galaxies.csv'
classification = pd.read_csv(classification_file, index_col=0).squeeze("columns").to_dict()

# === Kernel ===
def geodesic_kernel(r, cutoff=0.3):
    kern = np.exp(-r / cutoff)
    return kern / trapezoid(kern, r)

# === Model Functions ===
def model_geodesic(r, *params):
    M = params[0]
    return np.sqrt(M * np.convolve(geodesic_kernel(r), np.ones_like(r), 'same'))

def model_mond(r, M):
    a0 = 1.2e-10
    return np.sqrt((M * 1e-10) / r * (1 / (1 + (a0 * r / (M * 1e-10)))))

# === Metrics ===
def compute_metrics(y_true, y_pred, num_params):
    residual = y_true - y_pred
    chi2 = np.sum((residual)**2 / (y_true + 1e-5))
    dof = len(y_true) - num_params
    aic = chi2 + 2 * num_params
    bic = chi2 + num_params * np.log(len(y_true))
    return chi2 / dof, aic, bic

# === Directory ===
results = []
for file in os.listdir():
    if not file.endswith('.dat'): continue
    galaxy_name = file.replace('_rotmod.dat', '')
    galaxy_type = classification.get(galaxy_name, 'uncertain')

    try:
        data = np.loadtxt(file)
        radius = data[:, 0]
        velocity = data[:, 1]

        # Fit GEODESIC (trivial fit with dummy parameter)
        popt_geo, _ = curve_fit(lambda r, M: model_geodesic(r, M), radius, velocity, p0=[1])
        geo_pred = model_geodesic(radius, *popt_geo)
        geo_chi2, geo_aic, geo_bic = compute_metrics(velocity, geo_pred, len(popt_geo))

        # Fit MOND
        popt_mond, _ = curve_fit(model_mond, radius, velocity, p0=[1])
        mond_pred = model_mond(radius, *popt_mond)
        mond_chi2, mond_aic, mond_bic = compute_metrics(velocity, mond_pred, len(popt_mond))

        results.append({
            'Galaxy': galaxy_name,
            'Type': galaxy_type,
            'GEODESIC_chi2': geo_chi2,
            'GEODESIC_AIC': geo_aic,
            'GEODESIC_BIC': geo_bic,
            'MOND_chi2': mond_chi2,
            'MOND_AIC': mond_aic,
            'MOND_BIC': mond_bic
        })

    except Exception as e:
        print(f"❌ Failed for {galaxy_name}: {e}")

# === Save & Print ===
df = pd.DataFrame(results)
df.to_csv("grouped_model_comparison_results.csv", index=False)

grouped = df.groupby("Type").agg({
    'GEODESIC_chi2': ['mean', 'median'],
    'MOND_chi2': ['mean', 'median'],
    'GEODESIC_AIC': 'mean',
    'MOND_AIC': 'mean',
    'GEODESIC_BIC': 'mean',
    'MOND_BIC': 'mean'
})

print("\n📊 Grouped Summary by Galaxy Type:")
print(grouped)
