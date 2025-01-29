import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter

def latlonkm(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[:, 0])**2 + (coord1[1] - coord2[:, 1])**2)

def xymap(x, y, v=None, type='m', Nx=None, Ny=None, XLim=None, YLim=None):
    v = np.ones_like(x) if v is None else v
    if Nx is None or Ny is None:
        xn, yn = np.unique(x), np.unique(y)
        M = np.zeros((len(xn), len(yn)))
        for i in range(len(xn)):
            for j in range(len(yn)):
                M[i, j] = np.sum((x == xn[i]) & (y == yn[j]))
        return M, xn, yn
    
    minY, maxY = np.min(YLim if YLim is not None else y), np.max(YLim if YLim is not None else y)
    minX, maxX = np.min(XLim if XLim is not None else x), np.max(XLim if XLim is not None else x)
    I = np.where((x >= minX) & (x <= maxX) & (y >= minY) & (y <= maxY))
    x, y, v = x[I], y[I], v[I]
    
    stepx, stepy = (maxX - minX)/Nx, (maxY - minY)/Ny
    xn = np.maximum(np.ceil((x - minX)/stepx).astype(int), 1)
    yn = np.maximum(np.ceil((y - minY)/stepy).astype(int), 1)
    
    M = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            I = np.where((xn == i+1) & (yn == j+1))
            M[i, j] = np.nansum(v[I]) if type == 's' else np.nanmean(v[I])
    
    return M, np.arange(minX, maxX, stepx), np.arange(minY, maxY, stepy)

def bp(time, mag, Lon, Lat, depth, b, df, lag=None):
    lag = np.inf if lag is None else lag * 365.25 * 24 * 3600
    time = time * 365.25 * 24 * 3600
    P, n, T, D, M = np.zeros(len(time), dtype=int), np.full(len(time), np.inf), np.full(len(time), np.nan), np.full(len(time), np.nan), np.full(len(time), np.nan)
    P[np.argmin(time)] = 0
    
    for i in range(len(time)):
        I = np.where((time < time[i]) & (time >= time[i] - lag))[0]
        if I.size > 0:
            d = np.maximum(latlonkm([Lat[i], Lon[i]], np.array([Lat[I], Lon[I]]).T) * 1000, 0)
            t = np.maximum(time[i] - time[I], 0)
            nc = t * d**1.31 * 10**(-1.83 * mag[I])
            n[i], imin = np.min(nc), np.argmin(nc)
            P[i], D[i], T[i], M[i] = I[imin], d[imin], t[imin], mag[I[imin]]
    
    return P, n, D/(1000), T/(365.25 * 24 * 3600), M

def day_of_year(year, month, day):
    days = [31,29 if (year%4==0 and year%100!=0) or year%400==0 else 28,31,30,31,30,31,31,30,31,30,31]
    return sum(days[:month-1]) + day

# Main Script
cat = pd.read_csv('/TA/CODE/Main Python//MEQHULULAIS.txt', sep='\s+').values
year, month, day = cat[:,0].astype(int), cat[:,1].astype(int), cat[:,2].astype(int)
hour, minute, sec = cat[:,3].astype(int), cat[:,4].astype(int), cat[:,5].astype(float)
mag, Lon, Lat = cat[:,9].astype(float), cat[:,7].astype(float), cat[:,6].astype(float)

time = np.array([year[i] + (day_of_year(year[i],month[i],day[i]) + hour[i]/24 + minute[i]/1440 + sec[i]/86400)/365.25 for i in range(len(year))])

P, n, D, T, M = bp(time, mag, Lon, Lat, None, 1.0, 1.6)
TN = T * 10**(-0.5 * 1.83 * M)
RN = D**1.3 * 10**(-(1-0.5) * 1.83 * M)
eta_ij = TN * RN

I = np.where(~np.isnan(np.log10(eta_ij)) & (eta_ij > 0))[0]
eta0 = 10**np.mean(GaussianMixture(n_components=2).fit(np.log10(eta_ij[I]).reshape(-1,1)).means_.flatten())

plt.figure()
K, Ta, Xa = xymap(np.log10(TN), np.log10(RN), None, 's', 200, 200, [-12, 0], [-9, 0])

# plotting code
plt.pcolormesh(10**Ta, 10**Xa, gaussian_filter(K.T, sigma=2), shading='auto', cmap='viridis')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rescaled Time', fontsize=14)
plt.ylabel('Rescaled Distance', fontsize=14)
plt.title('TR Histogram')
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
plt.xlim(0, -1)
plt.ylim(0, 1)

Tlog = np.logspace(-11, 0, 100)
plt.loglog(Tlog, eta0/Tlog, 'k--', linewidth=2, label=f'η_0 = {eta0:.2e}', color = 'red')
plt.legend()
plt.show()
