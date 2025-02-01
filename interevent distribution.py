import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Tkagg")

file_path = "MEQHULULAIS.txt"
data = pd.read_csv(file_path, sep='\t', header=None, 
                   names=['year', 'month', 'day', 'hour', 'min', 'sec', 
                          'latitude', 'longitude', 'depth', 'mag'])

data["timestamp"] = pd.to_datetime(
    data["year"].astype(str) + "-" + data["month"].astype(str).str.zfill(2) + "-" +
    data["day"].astype(str).str.zfill(2) + " " + data["hour"].astype(str).str.zfill(2) + ":" +
    data["min"].astype(str).str.zfill(2) + ":" + data["sec"].astype(str)
)
data = data.sort_values("timestamp")

magnitudes = data["mag"].values
bins = np.arange(magnitudes.min(), magnitudes.max() + 0.1, 0.1)
hist, bin_edges = np.histogram(magnitudes, bins=bins)
Mc = bin_edges[np.argmax(hist)]
print(f"Mc yang dihitung dengan metode Maximum Curvature: {Mc:.2f}")

filtered_data = data[data["mag"] > Mc]
print(f"Jumlah data setelah filter Mc: {len(filtered_data)}")

if len(filtered_data) > 1:
    interevent_times = filtered_data["timestamp"].diff().dt.total_seconds().dropna()
    tau = interevent_times.mean()
    normalized_times = interevent_times / tau
    
    time_bins = np.logspace(np.log10(normalized_times.min()), np.log10(normalized_times.max()), 50)
    counts, bin_edges = np.histogram(normalized_times, bins=time_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    C, gamma, B = 0.5, 0.43, 1.58
    custom_pdf = C * bin_centers**(gamma - 1) * np.exp(-bin_centers / B)
    
    plt.figure(figsize=(6, 5))
    plt.plot(bin_centers, counts, 'o', label="Data")
    plt.plot(bin_centers, custom_pdf, label=f"gamma={gamma:.2f}, B={B:.2f}")
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(1e-4, 1e2)
    plt.ylim(1e-3, 1e3)
    
    plt.xlabel("Normalized Interevent Time (θ / τ)")
    plt.ylabel("Probability Density")
    plt.title(f"Interevent Time Distribution (Mc = {Mc:.2f})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
