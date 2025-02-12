import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

class FMD:
    def __init__(self):
        self.data = {'mag': np.array([]), 'magBins': np.array([]), 
                    'magHist': np.array([]), 'cumul': np.array([])}
        self.par = {'b': None, 'a': None, 'Mc': None, 'stdDev': None, 'binsize': 0.1}

    def mag_dist(self):
        self.data['mag'] = np.array(sorted(self.data['mag']))
        self.data['cumul'] = np.cumsum(np.ones(len(self.data['mag'])))[::-1]
        self.data['magBins'] = np.arange(round(self.data['mag'].min(), 1), 
                                       self.data['mag'].max() + self.par['binsize'],
                                       self.par['binsize'])
        self.data['magHist'], __ = np.histogram(self.data['mag'], self.data['magBins'])
        self.data['magBins'] = self.data['magBins'][0:-1] + self.par['binsize'] * .5

    def get_Mc(self, mc_type):
        if isinstance(mc_type, (np.ndarray, list)):
            self.par['Mc'] = self.Mc_KS(mc_type)
        else:
            self.mag_dist()
            sel = (self.data['magHist'] == self.data['magHist'].max())
            self.par['Mc'] = self.data['magBins'][sel.T].max()

    def KS_D_value_PL(self, Mc):
        aMag_tmp = np.sort(self.data['mag'][self.data['mag'] >= Mc])
        vX_tmp = 10 ** aMag_tmp
        xmin = 10 ** Mc
        n = aMag_tmp.shape[0]
        if n == 0: return np.inf
        alpha = float(n) / (np.log(vX_tmp / xmin)).sum()
        obsCumul = np.arange(n, dtype='float') / n
        modCumul = 1 - (xmin / vX_tmp) ** alpha
        return (abs(obsCumul - modCumul)).max()

    def Mc_KS(self, vMag_sim):
        sorted_Mag = np.sort(self.data['mag'])
        vMag_sim = np.sort(vMag_sim)
        vKS_stats = np.zeros(vMag_sim.shape[0])
        
        for i, curr_Mc in enumerate(vMag_sim):
            vKS_stats[i] = self.KS_D_value_PL(curr_Mc)
        
        self.data['a_KS'] = vKS_stats
        self.data['a_MagSim'] = vMag_sim
        return vMag_sim[vKS_stats == vKS_stats.min()][0]

    def fit_GR(self, binCorrection=0):
        sel_Mc = self.data['mag'] >= self.par['Mc']
        N = sel_Mc.sum()
        meanMag = self.data['mag'][sel_Mc].mean()
        self.par['b'] = (1 / (meanMag - (self.par['Mc'] - binCorrection))) * np.log10(np.e)
        self.par['stdDev'] = (2.3 * np.sqrt((sum((self.data['mag'][sel_Mc] - meanMag) ** 2)) / 
                            (N * (N - 1)))) * self.par['b'] ** 2
        self.par['a'] = np.log10(N) + self.par['b'] * self.par['Mc']

    def plotFit(self, ax):
        N = len(self.data['mag'][self.data['mag'] >= self.par['Mc']])
        ax.semilogy(self.data['magBins'], self.data['magHist'], 'ks', ms=5, mew=1, label='histogram')
        sel = self.data['mag'] > self.par['Mc'] - 1
        ax.semilogy(self.data['mag'][sel], self.data['cumul'][sel], 'bo', ms=2, label='cumulative')
        
        sel = abs(self.data['mag'] - self.par['Mc']) == abs(self.data['mag'] - self.par['Mc']).min()
        ax.plot([self.par['Mc']], [self.data['cumul'][sel][0]], 'rv', ms=4,
                label=f"$M_c = {round(self.par['Mc'], 1)}$")

        mag_hat = np.linspace(self.data['mag'].min() - 2 * self.par['binsize'],
                            self.data['mag'].max() + 2 * self.par['binsize'], 10)
        N_hat = 10 ** ((-self.par['b'] * mag_hat) + self.par['a'])
        ax.semilogy(mag_hat, N_hat, 'r--',
                   label='$log(N) = -%.1f \\cdot M + %.1f$' % (round(self.par['b'], 1), round(self.par['a'], 1)))

        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Number of Events')
        ax.set_title('$N (M>M_c) = %.0f ; \\; \\sigma_b = %.3f $' % (N, self.par['stdDev']))
        ax.legend(shadow=False, numpoints=1, loc='upper right')
        ax.set_ylim(1, len(self.data['mag']) * 1.2)
        ax.grid('on')

    def plotKS(self, ax):
        if 'a_KS' in self.data:
            ax.plot(self.data['a_MagSim'], self.data['a_KS'], 'b-')
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('KS-D')

if __name__ == "__main__":
    np.random.seed(123456)
    oFMD = FMD()
    
    # Load and process data
    dir_in = 'data'
    file_in = 'MEQHULULAISALL.mat'
    dEq = scipy.io.loadmat(f"{dir_in}/{file_in}", struct_as_record=False, squeeze_me=True)
    oFMD.data['mag'] = dEq['Mag']
    print('no of events', len(dEq['Mag']))
    
    # Analysis
    binsize = 0.1
    mc_type = np.arange(0.0, 2.0, binsize)
    a_RanErr = np.random.randn(len(dEq['Mag'])) * binsize * .4
    oFMD.data['mag'] += a_RanErr
    oFMD.mag_dist()
    oFMD.get_Mc(mc_type)
    oFMD.data['mag'] -= a_RanErr
    
    print('completeness', round(oFMD.par['Mc'], 1))
    oFMD.fit_GR()
    print(oFMD.par)
    
    # Plotting
    plt.figure(1, figsize=(6, 10))
    ax1 = plt.subplot(211)
    oFMD.plotFit(ax1)
    ax2 = plt.subplot(212)
    oFMD.plotKS(ax2)
    ax2.set_xlim(ax1.get_xlim())
    plt.savefig(file_in.replace('.mat', '_fmd.png'))
    plt.show()