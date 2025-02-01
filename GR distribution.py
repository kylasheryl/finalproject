import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use("Tkagg")

def plot_gr_distribution(filepath):
    data = pd.read_csv(filepath, sep='\t', header=None, 
                       names=['year', 'month', 'day', 'hour', 'min', 'sec', 
                              'latitude', 'longitude', 'depth', 'mag'])
    
    magnitudes = data['mag'].values
    
    bins = np.arange(magnitudes.min(), magnitudes.max() + 0.1, 0.1)
    hist, bin_edges = np.histogram(magnitudes, bins=bins)
    mc_value = bin_edges[np.argmax(hist)]
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    cum_freq = np.array([np.sum(magnitudes >= mag) for mag in bin_centers])
    
    magnitudes_above_mc = magnitudes[magnitudes >= mc_value]
    if len(magnitudes_above_mc) > 0:
        b_value = np.log10(np.e) / (np.mean(magnitudes_above_mc) - mc_value)
        a_value = np.log10(len(magnitudes_above_mc)) + b_value * mc_value
    else:
        b_value, a_value = np.nan, np.nan
    
    plt.figure(figsize=(7, 6))
    plt.plot(bin_centers, hist, 'bs', label='Magnitude-frequency', markersize=3)
    plt.plot(bin_centers, cum_freq, 'ro', label='Cumulative magnitude-frequency', markersize=3)
    
    if not np.isnan(b_value):
        plt.plot(bin_centers, 10**(a_value - b_value * bin_centers), 'k-', 
                 label=f'GR law (b = {b_value:.4f})')
    plt.axvline(x=mc_value, color='g', linestyle='--', label=f'Mc = {mc_value:.2f}')
    
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency log(N)')
    plt.title('Magnitude-Frequency Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {'a_value': a_value, 'b_value': b_value, 'mc_value': mc_value}

file_path = "MEQHULULAIS.txt"
results = plot_gr_distribution(file_path)
print(f"\nHasil Analisis:")
print(f"a-value = {results['a_value']:.4f}")
print(f"b-value = {results['b_value']:.4f}")
print(f"Magnitude of Completeness (Mc) = {results['mc_value']:.2f}")
