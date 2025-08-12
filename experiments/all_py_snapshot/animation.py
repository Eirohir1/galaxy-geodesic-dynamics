import matplotlib.pyplot as plt
import numpy as np

# Data from your output
galaxy_types = ['dwarf', 'spiral', 'super-spiral', 'uncertain']

geodesic_chi2 = [14.90, 52.81, 102.24, 33.61]
mond_chi2 = [18.96, 62.64, 104.75, 41.84]

geodesic_aic = [133.56, 917.02, 3129.51, 712.22]
mond_aic = [182.57, 1147.96, 3644.37, 838.71]

geodesic_bic = [133.63, 917.70, 3130.80, 712.56]
mond_bic = [182.63, 1148.64, 3645.65, 839.05]

x = np.arange(len(galaxy_types))
width = 0.35

fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Chi-squared
axs[0].bar(x - width/2, geodesic_chi2, width, label='GEODESIC')
axs[0].bar(x + width/2, mond_chi2, width, label='MOND')
axs[0].set_ylabel('Mean Chi-Squared')
axs[0].set_title('Mean Chi-Squared by Galaxy Type')
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.6)

# AIC
axs[1].bar(x - width/2, geodesic_aic, width, label='GEODESIC')
axs[1].bar(x + width/2, mond_aic, width, label='MOND')
axs[1].set_ylabel('Mean AIC')
axs[1].set_title('Mean Akaike Information Criterion by Galaxy Type')
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.6)

# BIC
axs[2].bar(x - width/2, geodesic_bic, width, label='GEODESIC')
axs[2].bar(x + width/2, mond_bic, width, label='MOND')
axs[2].set_ylabel('Mean BIC')
axs[2].set_title('Mean Bayesian Information Criterion by Galaxy Type')
axs[2].set_xticks(x)
axs[2].set_xticklabels(galaxy_types)
axs[2].legend()
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.xlabel('Galaxy Type')
plt.tight_layout()
plt.show()
