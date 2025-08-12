import matplotlib.pyplot as plt
import numpy as np

# Success rates from your breakthrough analysis
galaxy_types = ['Dwarf', 'Spiral', 'Massive', 'Diffuse', 'Barred']
success_rates = [97.8, 90.9, 24.3, 100.0, 71.4]
galaxy_counts = [44, 50, 9, 1, 5]
colors = ['green', 'blue', 'red', 'purple', 'orange']

plt.figure(figsize=(12, 8))
bars = plt.bar(galaxy_types, success_rates, color=colors, alpha=0.7)
plt.ylabel('Success Rate (%%)', fontsize=14)
plt.xlabel('Galaxy Type', fontsize=14)
plt.title('Geodesic Theory: 76.1%% of Galaxies Explained Without Dark Matter', fontsize=16)
plt.axhline(y=76.1, color='black', linestyle='--', linewidth=2, label='Overall Success: 76.1%%')
for bar, count in zip(bars, galaxy_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{count} galaxies', ha='center', fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/success_by_galaxy_type.png', dpi=300, bbox_inches='tight')
print('Summary figure saved!')
