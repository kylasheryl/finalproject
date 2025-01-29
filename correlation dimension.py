import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

data = np.loadtxt("MEQHULULAIS.txt", skiprows=1)
coords = np.radians(data[:, 6:8])
distances = 2 * 6371 * np.arcsin(np.clip(pdist(coords) / 2, -1, 1))

r_min, r_max = np.log10(np.min(distances[distances > 0])), np.log10(np.max(distances))
r_bins = np.linspace(r_min, r_max, 100)

N = len(distances)
C_r = [(2.0 * np.sum(distances <= 10**r)) / (N * (N-1)) for r in r_bins]
d_corr = np.gradient(np.log10(np.clip(C_r, 1e-10, None)), r_bins)

print(f"Mean Correlation Dimension: {np.mean(d_corr):.3f}")
print(f"Max Correlation Dimension: {np.max(d_corr):.3f}")

plt.figure(figsize=(7, 6))
plt.grid(True, alpha=0.3)
plt.plot(r_bins, d_corr, 'r-', label='Correlation Dimension', linewidth=2)
plt.axvline(x=r_min, color='blue', linestyle='--', label=f'r_min = {r_min:.2f}', zorder=5)
plt.axvline(x=r_max, color='green', linestyle='--', label=f'r_max = {r_max:.2f}', zorder=5)
plt.xlabel('log(r [km])')
plt.ylabel('Correlation Dimension')
plt.title('Correlation Dimension')
plt.legend()
plt.ylim(-0.5, 6)
plt.xlim(r_min - 0.1, r_max + 0.1)
plt.tight_layout()
plt.show()