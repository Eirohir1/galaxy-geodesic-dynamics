import numpy as np

# Step 1: Load curvature field from your existing CSV (adjust if needed)
csv_file = 'bullet_cluster_phase3_gradient_field.csv'
curvature = np.loadtxt(csv_file, delimiter=',')

# Step 2: Save as .npy for universe evolution
npy_file = 'curvature_map.npy'
np.save(npy_file, curvature)

print(f"✅ Saved curvature map to: {npy_file}")
