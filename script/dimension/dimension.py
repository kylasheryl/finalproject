import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import matplotlib 
matplotlib.use("Tkagg")

data = np.loadtxt("MEQHULULAISALL.txt", skiprows=1)
coords = np.radians(data[:, 6:8])
distances = 2 * 6371 * np.arcsin(np.clip(pdist(coords) / 2, -1, 1))

r_min, r_max = np.log10(np.min(distances[distances > 0.01])), np.log10(np.max(distances))
r_bins = np.linspace(r_min, r_max, 100)

N = len(distances)
C_r = [(2.0 * np.sum(distances <= 10**r)) / (N * (N-1)) for r in r_bins]
d_corr = np.gradient(np.log10(np.clip(C_r, 1e-10, None)), r_bins)

print(f"Mean Correlation Dimension: {np.mean(d_corr):.3f}")
print(f"Max Correlation Dimension: {np.max(d_corr):.3f}")

plt.figure(figsize=(7, 6))
plt.grid(True, alpha=0.3)
plt.plot(r_bins, d_corr, 'ro', label='Correlation Dimension', linewidth=2)
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


def compute_correlation_sum(coords, r_values):
    """
    Compute the correlation sum C(r) for a set of points.

    Parameters:
        coords (np.array): An array of shape (N, D) containing N points in D-dimensional space.
        r_values (np.array): Array of r values at which to compute C(r).

    Returns:
        np.array: The correlation sum values for each r.
    """
    N = coords.shape[0]
    # Compute all unique pairwise distances
    distances = pdist(coords)
    total_pairs = N * (N - 1) / 2.0

    # For each r, count the fraction of pairs with distance less than r.
    C_r = np.array([np.sum(distances < r) / total_pairs for r in r_values])
    return C_r

def spherical_to_cartesian(lat, lon, depth, R=6371.0):
    """
    Convert geographic coordinates (latitude, longitude, depth) to Cartesian coordinates.
    
    Parameters:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        depth (float): Depth in km (measured from the surface).
        R (float): Earth's radius in km (default: 6371 km).
        
    Returns:
        np.array: Cartesian coordinates [x, y, z].
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    r = R - depth  # effective distance from the Earth's center
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return np.array([x, y, z])

event_data = data[:,6:9]

# Convert list of [lat, lon, depth] to list of tuples if necessary
events = [tuple(item) for item in event_data]

# Convert events into 3D Cartesian coordinates
coords = np.array([spherical_to_cartesian(lat, lon, depth) for lat, lon, depth in events])

# Determine a range for r values (logarithmically spaced)
# Compute all pairwise distances to set appropriate r_min and r_max
all_distances = pdist(coords)
nonzero_distances = all_distances[all_distances > 0]
r_min = np.min(nonzero_distances)
r_max = np.max(all_distances)
r_values = np.logspace(np.log10(r_min), np.log10(r_max), 100)

# Compute the correlation sum C(r)
C_r = compute_correlation_sum(coords, r_values)

# Compute logarithms
log_r = np.log(r_values)
log_C = np.log(C_r)

# Compute the local slope d(log(C))/d(log(r)) which estimates the local correlation dimension
# Here we use a numerical gradient.
local_correlation_dimension = np.gradient(log_C, log_r)

# -------------------------
# Plotting
# -------------------------
plt.figure(figsize=(8, 6))
plt.plot(log_r, local_correlation_dimension, 'o-', label='Local correlation dimension')
plt.xlabel('log(r)')
plt.ylabel('d(log(C))/d(log(r))')
plt.title('Local Correlation Dimension vs log(r)')
plt.legend()
plt.show()