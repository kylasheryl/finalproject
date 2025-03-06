import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.io
from scipy.spatial.distance import pdist
from src.EqCat import EqCat
from src import clustering, data_utils

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
        if N == 0:
            self.par['b'] = 1.0  # Default value if no events
            self.par['stdDev'] = 0.0
            self.par['a'] = 0.0
            return
            
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
        ax.grid(True)

    def plotKS(self, ax):
        if 'a_KS' in self.data:
            ax.plot(self.data['a_MagSim'], self.data['a_KS'], 'b-')
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('KS-D')
            ax.axvline(x=self.par['Mc'], color='r', linestyle='--', label=f'Mc = {self.par["Mc"]:.2f}')
            ax.grid(True)
            ax.legend()

def compute_correlation_dimension(coords):
    """
    Compute fractal dimension using method from dimension.py
    
    Parameters:
        coords (np.array): Coordinates (x, y) of events
    
    Returns:
        float: Fractal dimension
        np.array: r values for plotting
        np.array: C(r) values for plotting
        np.array: d_corr values for plotting
    """
    # Check if we have sufficient data
    if len(coords) < 10:
        return 1.6, np.logspace(0, 3, 30), np.logspace(-3, 0, 30), np.ones(30) * 1.6
    
    # Compute pairwise distances
    try:
        # Convert to radians if the data seems to be in geographic coordinates
        if np.max(np.abs(coords)) <= 180:
            coords_rad = np.radians(coords[:, :2])  # Use only first two columns (lat, lon)
            # Use great circle distance as in dimension.py
            distances = 2 * 6371 * np.arcsin(np.clip(pdist(coords_rad) / 2, -1, 1))
        else:
            # Use Euclidean distance for cartesian coordinates
            distances = pdist(coords)
        
        # Remove zero distances and very small distances
        nonzero_distances = distances[distances > 0.01]
        
        if len(nonzero_distances) < 10:
            return 1.6, np.logspace(0, 3, 30), np.logspace(-3, 0, 30), np.ones(30) * 1.6
        
        # Get range for r values on log scale EXACTLY as in dimension.py
        r_min = np.log10(np.min(distances[distances > 0.01]))
        r_max = np.log10(np.max(distances))
        
        # Create r_bins with logarithmic scale EXACTLY as in dimension.py
        r_bins = np.linspace(r_min, r_max, 100)
        
        # Calculate correlation integral EXACTLY as in dimension.py
        N = len(distances)
        C_r = [(2.0 * np.sum(distances <= 10**r)) / (N * (N-1)) for r in r_bins]
        
        # Calculate correlation dimension EXACTLY as in dimension.py
        d_corr = np.gradient(np.log10(np.clip(C_r, 1e-10, None)), r_bins)
        
        # Calculate mean correlation dimension
        D = np.mean(d_corr)
        
        # Get r_values for plotting (in original units, not log)
        r_values = 10**r_bins
        
        return D, r_values, C_r, d_corr
    
    except Exception as e:
        print(f"Error in compute_correlation_dimension: {e}")
        return 1.6, np.logspace(0, 3, 30), np.logspace(-3, 0, 30), np.ones(30) * 1.6

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
    if file_path:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, file_path)
        status_var.set("File loaded successfully")
        status_label.config(fg="green")
        
        # Auto calculate Mc, b-value, and D after loading file if auto calculate is checked
        if auto_calculate_var.get():
            calculate_parameters()

def clear_output_plots():
    # Clear all tabs in the notebook
    for widget in output_notebook.tabs():
        output_notebook.forget(widget)

def calculate_parameters():
    file_path = entry_file.get()
    if not file_path:
        status_var.set("Pilih file katalog gempa terlebih dahulu!")
        status_label.config(fg="red")
        return
    
    try:
        # Create EqCat object
        eqCat = EqCat()
        eqCat.loadMatBin(file_path)
        status_var.set(f"Total events: {eqCat.size()}")
        status_label.config(fg="blue")
        root.update()
        
        # Create progress window
        progress_window = tk.Toplevel(root)
        progress_window.title("Calculating Parameters")
        progress_label = tk.Label(progress_window, text="Calculating Mc, b-value, and D...")
        progress_label.pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, length=300, mode='determinate')
        progress_bar.pack(pady=10)
        progress_var.set(10)
        progress_window.update()
        
        # ---------------------------------------------------------
        # Calculate Mc and b-value using improved FMD class
        # ---------------------------------------------------------
        magnitudes = eqCat.data['Mag']
        
        # Create FMD object
        oFMD = FMD()
        oFMD.data['mag'] = magnitudes
        
        # Add small random perturbation to magnitudes
        binsize = 0.1
        a_RanErr = np.random.randn(len(magnitudes)) * binsize * 0.4
        oFMD.data['mag'] += a_RanErr
        
        # Calculate magnitude distribution
        oFMD.mag_dist()
        
        # Calculate Mc using KS method
        mc_type = np.arange(np.min(magnitudes), np.max(magnitudes), binsize)
        if len(mc_type) < 2:  # Ensure we have at least 2 points
            mc_type = np.array([np.min(magnitudes), np.max(magnitudes)])
        oFMD.get_Mc(mc_type)
        
        # Restore original magnitudes and fit G-R law
        oFMD.data['mag'] -= a_RanErr
        oFMD.fit_GR()
        
        # Get calculated values
        Mc = oFMD.par['Mc']
        b_value = oFMD.par['b']
        
        progress_var.set(50)
        progress_window.update()
        
        # ---------------------------------------------------------
        # Calculate fractal dimension
        # ---------------------------------------------------------
        # First convert to cartesian coordinates
        eqCat.toCart_coordinates(projection='eqdc')
        
        # Extract coordinates
        coords = np.column_stack((eqCat.data['X'], eqCat.data['Y']))
        
        # Calculate fractal dimension with improved method
        D, r_values, C_r, d_corr = compute_correlation_dimension(coords)
        
        progress_var.set(100)
        progress_window.update()
        
        # Close progress window
        progress_window.destroy()
        
        # Update entry fields with calculated values
        entry_Mc.delete(0, tk.END)
        entry_Mc.insert(0, f"{Mc:.2f}")
        
        entry_b.delete(0, tk.END)
        entry_b.insert(0, f"{b_value:.2f}")
        
        entry_D.delete(0, tk.END)
        entry_D.insert(0, f"{D:.2f}")
        
        # Clear previous plots
        clear_output_plots()
        
        # Show plot in the output area
        update_parameter_plots(eqCat, magnitudes, Mc, b_value, D, oFMD, r_values, C_r, d_corr)
        
    except Exception as e:
        status_var.set(f"Error calculating parameters: {str(e)}")
        status_label.config(fg="red")
        messagebox.showerror("Error", str(e))

def update_parameter_plots(eqCat, magnitudes, Mc, b_value, D, oFMD, r_values, C_r, d_corr):
    # Clear previous tabs
    for widget in output_notebook.tabs():
        output_notebook.forget(widget)
    
    # Create Mc and b-value plot tab using plotFit
    tab1 = ttk.Frame(output_notebook)
    output_notebook.add(tab1, text='Magnitude Distribution')
    
    fig1 = Figure(figsize=(6, 4))
    ax1 = fig1.add_subplot(111)
    
    # Plot using FMD plotting function
    oFMD.plotFit(ax1)
    fig1.tight_layout()
    
    canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Create KS statistic plot if available
    if 'a_KS' in oFMD.data:
        tab2 = ttk.Frame(output_notebook)
        output_notebook.add(tab2, text='KS Statistics')
        
        fig2 = Figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        
        # Plot KS statistics
        oFMD.plotKS(ax2)
        fig2.tight_layout()
        
        canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Create fractal dimension plot tab - EXACTLY like dimension.py
    tab3 = ttk.Frame(output_notebook)
    output_notebook.add(tab3, text='Fractal Dimension')
    
    # Get log values 
    log_r = np.log10(r_values)
    
    # Get r_min and r_max for display
    r_min = log_r[0]
    r_max = log_r[-1]
    
    fig3 = Figure(figsize=(6, 4))
    ax3 = fig3.add_subplot(111)
    
    # Plot dimension vs log(r) 
    ax3.grid(True, alpha=0.3)
    ax3.plot(log_r, d_corr, 'ro', label='Correlation Dimension')
    
    ax3.axhline(y=D, color='blue', linestyle='-', linewidth=2, label=f'D = {D:.2f}', zorder=10)
    
    ax3.axvline(x=r_min, color='blue', linestyle='--', label=f'r_min = {r_min:.2f}', zorder=5)
    ax3.axvline(x=r_max, color='green', linestyle='--', label=f'r_max = {r_max:.2f}', zorder=5)
    ax3.set_xlabel('log(r [km])')
    ax3.set_ylabel('Correlation Dimension')
    ax3.set_title(f'Correlation Dimension (D = {D:.2f})')
    ax3.legend()
    ax3.set_ylim(-0.5, 6)
    ax3.set_xlim(r_min - 0.1, r_max + 0.1)
    fig3.tight_layout()
    
    canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Set first tab as active
    output_notebook.select(0)

def run_analysis():
    file_path = entry_file.get()
    if not file_path:
        status_var.set("Pilih file katalog gempa terlebih dahulu!")
        status_label.config(fg="red")
        return
    
    try:
        # Parameters
        M_c = float(entry_Mc.get())
        b = float(entry_b.get())
        D = float(entry_D.get())
        rmax = float(entry_rmax.get())  # Manual rmax value
        tmax = float(entry_tmax.get())  # Manual tmax value
        Tmin, Tmax = -10, 0
        Rmin, Rmax = -5, 3
        nBoot = int(entry_nBoot.get())  # Number of bootstraps
        
        # Get custom suffix if provided
        custom_suffix = entry_suffix.get()
        if custom_suffix and not custom_suffix.startswith('_'):
            custom_suffix = '_' + custom_suffix
        
        # Create EqCat objects
        eqCat = EqCat()
        eqCat.loadMatBin(file_path)
        status_var.set(f"Total events: {eqCat.size()}")
        status_label.config(fg="blue")
        root.update()
        
        # Select events above magnitude cutoff
        eqCat.selectEvents(M_c, None, 'Mag')
        eqCat.toCart_coordinates(projection='eqdc')
        status_var.set(f"Events after selection: {eqCat.size()}")
        root.update()
        
        # Create constant dictionary
        dConst = {'Mc': M_c, 'b': b, 'D': D}
        
        # Calculate NND for original catalog with manual rmax and tmax
        dNND = clustering.NND_eta(eqCat, dConst, correct_co_located=True, verbose=False, 
                               rmax=rmax, tmax=tmax)
        
        # Calculate eta_0 using percentile method
        eta_0_values = np.zeros(nBoot)
        
        # Create progress window
        progress_window = tk.Toplevel(root)
        progress_window.title("Processing")
        progress_label = tk.Label(progress_window, text="Calculating eta_0 values...")
        progress_label.pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, length=300, mode='determinate')
        progress_bar.pack(pady=10)
        
        # Update main window
        root.update()
        
        # Randomized catalog process
        ranCat = EqCat()
        for i in range(nBoot):
            # Update progress
            progress_var.set((i+1) / nBoot * 100)
            progress_label.config(text=f"Processing bootstrap {i+1}/{nBoot}...")
            progress_window.update()
            
            # Create randomized catalog
            ranCat.copy(eqCat)
            ranCat.data['X'] = np.random.uniform(eqCat.data['X'].min(), eqCat.data['X'].max(), size=eqCat.size())
            ranCat.data['Y'] = np.random.uniform(eqCat.data['Y'].min(), eqCat.data['Y'].max(), size=eqCat.size())
            ranCat.data['Time'] = clustering.rand_rate_uni(eqCat.size(), eqCat.data['Time'].min(), eqCat.data['Time'].max())
            ranCat.sortCatalog('Time')
            
            # Compute NND for randomized catalog
            dNND_ran = clustering.NND_eta(ranCat, dConst, correct_co_located=True, verbose=False,
                                       rmax=rmax, tmax=tmax)
            
            # Calculate eta_0 using 1st percentile
            eta_0_values[i] = round(np.percentile(np.log10(dNND_ran['aNND']), 1), 5)
        
        # Close progress window
        progress_window.destroy()
        
        # Calculate mean eta_0
        eta_0 = eta_0_values.mean()
        
        # Save results to file
        output_dir = os.path.dirname(file_path)
        output_base = os.path.basename(file_path).replace('.mat', '')
        
        # Add custom suffix to filenames if provided
        if custom_suffix:
            output_nnd_file = os.path.join(output_dir, f"{output_base}_NND{custom_suffix}.mat")
            output_eta_file = os.path.join(output_dir, f"{output_base}_Mc_{M_c}{custom_suffix}_eta_0.txt")
            output_rt_file = os.path.join(output_dir, f"{output_base}_RT_Mc_{M_c}{custom_suffix}.mat")
        else:
            output_nnd_file = os.path.join(output_dir, f"{output_base}_NND.mat")
            output_eta_file = os.path.join(output_dir, f"{output_base}_Mc_{M_c}_eta_0.txt")
            output_rt_file = os.path.join(output_dir, f"{output_base}_RT_Mc_{M_c}.mat")
        
        # Save NND data
        scipy.io.savemat(output_nnd_file, dNND, do_compression=True)
        
        # Save eta_0 data
        np.savetxt(output_eta_file, np.array([eta_0]), fmt='%10.3f', header='eta_0')
        
        eta_mat_file = output_eta_file.replace('txt', 'mat')
        scipy.io.savemat(eta_mat_file, 
                         {'eta_0': eta_0, 'eta_BS': eta_0_values}, do_compression=True)
        
        # Create child and parent catalogs for R-T plot
        catChild = EqCat()
        catParent = EqCat()
        catChild.copy(eqCat)
        catParent.copy(eqCat)
        catChild.selEventsFromID(dNND['aEqID_c'], repeats=True)
        catParent.selEventsFromID(dNND['aEqID_p'], repeats=True)
        
        # Calculate rescaled time and distance
        a_R, a_T = clustering.rescaled_t_r(catChild, catParent, dConst, correct_co_located=True)
        
        # Store the pairs of parent and child IDs corresponding to a_R and a_T
        parent_child_pairs = {
            'parent_ids': catParent.data['N'],  # IDs from the parent catalog
            'child_ids': catChild.data['N'],    # IDs from the child catalog
            'rescaled_time': a_T,               # Rescaled time values
            'rescaled_distance': a_R            # Rescaled distance values
        }
        
        # Define bins for 2D density
        a_Tbin = np.arange(Tmin, Tmax+0.2, 0.1)
        a_Rbin = np.arange(Rmin, Rmax+0.2, 0.1)
        
        # Create R-T density
        XX, YY, ZZ = data_utils.density_2D(np.log10(a_T), np.log10(a_R), a_Tbin, a_Rbin, sigma=None)
        
        # Save R-T data to mat file
        scipy.io.savemat(output_rt_file, {
            'XX': XX,
            'YY': YY,
            'ZZ': ZZ,
            'eta_0': eta_0,
            'log_T': np.log10(a_T),
            'log_R': np.log10(a_R),
            'T': a_T,  # Original T values
            'R': a_R,  # Original R values
            'parent_ids': parent_child_pairs['parent_ids'],  # Parent IDs
            'child_ids': parent_child_pairs['child_ids'],   # Child IDs
            'Tmin': Tmin,
            'Tmax': Tmax,
            'Rmin': Rmin,
            'Rmax': Rmax
        }, do_compression=True)
        
        # Clear output tabs and add new result tabs
        clear_output_plots()
        
        # Add NND histogram tab
        nnd_tab = ttk.Frame(output_notebook)
        output_notebook.add(nnd_tab, text='NND Histogram')
        
        fig_nnd = Figure(figsize=(6, 4))
        ax_nnd = fig_nnd.add_subplot(111)
        bins = np.arange(-13, 0, 0.3)  # Similar to the original script
        ax_nnd.hist(np.log10(dNND['aNND']), bins, color='.5', align='mid', rwidth=0.9)
        
        # Plot eta_0 line (white solid line)
        ax_nnd.plot([eta_0, eta_0], ax_nnd.get_ylim(), 'w-', lw=2)
        # Plot eta_0 line (red dashed line)
        ax_nnd.plot([eta_0, eta_0], ax_nnd.get_ylim(), 'r--', lw=2, 
                label=f'η₀ = {eta_0:.3f}, N_total={eqCat.size()}')
        
        # Add text with eta_0 value
        ax_nnd.text(eta_0 + 0.5, ax_nnd.get_ylim()[1] * 0.9, f'η₀ = {eta_0:.3f}', 
                color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        ax_nnd.legend(loc='upper left')
        ax_nnd.set_xlabel('NND, log₁₀ η')
        ax_nnd.set_ylabel('Number of Events')
        ax_nnd.grid(True)
        ax_nnd.set_xlim(-13, 0)
        ax_nnd.set_title(f"Distribusi log₁₀(η) untuk Clustering Earthquake (Mc={M_c})")
        fig_nnd.tight_layout()
        
        canvas_nnd = FigureCanvasTkAgg(fig_nnd, master=nnd_tab)
        canvas_nnd.draw()
        canvas_nnd.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add R-T plot tab
        rt_tab = ttk.Frame(output_notebook)
        output_notebook.add(rt_tab, text='R-T Plot')
        
        fig_rt = Figure(figsize=(6, 5))
        ax_rt = fig_rt.add_subplot(111)
        ax_rt.set_title('Nearest Neighbor Pairs in R-T')
        
        # Plot density
        plot1 = ax_rt.pcolormesh(XX, YY, ZZ, cmap=plt.cm.RdYlGn_r)
        cbar = fig_rt.colorbar(plot1, orientation='horizontal', shrink=0.5, aspect=20)
        
        # Plot eta_0 line (white solid)
        ax_rt.plot([Tmin, Tmax], -np.array([Tmin, Tmax])+eta_0, '-', lw=1.5, color='w')
        # Plot eta_0 line (gray dashed)
        ax_rt.plot([Tmin, Tmax], -np.array([Tmin, Tmax])+eta_0, '--', lw=1.5, color='.5')
        
        # Add text with eta_0 value
        ax_rt.text(Tmin + 1, Rmin + 1, f'η₀ = {eta_0:.3f}', 
                color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
        
        # Labels and legends
        cbar.set_label('Number of Event Pairs', labelpad=-40)
        ax_rt.set_xlabel('Rescaled Time')
        ax_rt.set_ylabel('Rescaled Distance')
        ax_rt.set_xlim(Tmin, Tmax)
        ax_rt.set_ylim(Rmin, Rmax)
        fig_rt.tight_layout()
        
        canvas_rt = FigureCanvasTkAgg(fig_rt, master=rt_tab)
        canvas_rt.draw()
        canvas_rt.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Select first tab to display
        output_notebook.select(0)
        
        # Update status - using StringVar and messagebox for results
        status_var.set(f"Analisis selesai! η₀={eta_0:.3f}")
        status_label.config(fg="green")
        
        # Show a separate message box with file paths
        messagebox.showinfo("Output Files", 
                           f"Hasil disimpan di:\n\n1. {output_nnd_file}\n\n2. {output_eta_file}\n\n3. {output_rt_file}")
        
    except Exception as e:
        status_var.set(f"Error: {str(e)}")
        status_label.config(fg="red")
        messagebox.showerror("Error", str(e))

# Create GUI
root = tk.Tk()
root.title("Earthquake Clustering Analysis")
root.geometry("1200x700")  # Increased width for side-by-side layout

# Create main frame with two columns
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left frame for inputs
left_frame = ttk.Frame(main_frame, padding=(5, 5, 5, 5))
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

# Right frame for outputs
right_frame = ttk.LabelFrame(main_frame, text="Results and Visualizations", padding=(5, 5, 5, 5))
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

# File selection
frame_file = ttk.LabelFrame(left_frame, text="File Selection", padding=(5, 5, 5, 5))
frame_file.pack(pady=5, fill=tk.X)

ttk.Label(frame_file, text="Pilih File Katalog .mat").grid(row=0, column=0, sticky=tk.W, pady=2)
entry_file = ttk.Entry(frame_file, width=30)
entry_file.grid(row=0, column=1, padx=5, pady=2)
ttk.Button(frame_file, text="Browse", command=load_file).grid(row=0, column=2, padx=5, pady=2)

# Auto calculate option
auto_calculate_var = tk.BooleanVar()
auto_calculate_var.set(True)  # Default to auto calculate
auto_calc_check = ttk.Checkbutton(frame_file, text="Auto Calculate on Load", variable=auto_calculate_var)
auto_calc_check.grid(row=1, column=1, sticky=tk.W, pady=2)

# Parameters frame
frame_params = ttk.LabelFrame(left_frame, text="Parameter Estimation", padding=(5, 5, 5, 5))
frame_params.pack(pady=5, fill=tk.X)

# M_c (Magnitude Cutoff)
ttk.Label(frame_params, text="Mc (Magnitude Cutoff)").grid(row=0, column=0, sticky=tk.W, pady=5)
entry_Mc = ttk.Entry(frame_params, width=10)
entry_Mc.grid(row=0, column=1, sticky=tk.W, padx=5)
entry_Mc.insert(0, "")

# b (b-value)
ttk.Label(frame_params, text="b-value").grid(row=1, column=0, sticky=tk.W, pady=5)
entry_b = ttk.Entry(frame_params, width=10)
entry_b.grid(row=1, column=1, sticky=tk.W, padx=5)
entry_b.insert(0, "")

# D (Fractal Dimension)
ttk.Label(frame_params, text="D (Fractal Dimension)").grid(row=2, column=0, sticky=tk.W, pady=5)
entry_D = ttk.Entry(frame_params, width=10)
entry_D.grid(row=2, column=1, sticky=tk.W, padx=5)
entry_D.insert(0, "")

# Button to calculate parameters
ttk.Button(frame_params, text="Calculate Parameters", command=calculate_parameters).grid(row=3, column=0, columnspan=2, pady=5)

# Additional parameters
frame_params2 = ttk.LabelFrame(left_frame, text="Clustering Parameters", padding=(5, 5, 5, 5))
frame_params2.pack(pady=5, fill=tk.X)

# rmax (Maximum distance in km)
ttk.Label(frame_params2, text="rmax (km)").grid(row=0, column=0, sticky=tk.W, pady=5)
entry_rmax = ttk.Entry(frame_params2, width=10)
entry_rmax.grid(row=0, column=1, sticky=tk.W, padx=5)
entry_rmax.insert(0, "50")  # Default value

# tmax (Maximum time in years)
ttk.Label(frame_params2, text="tmax (years)").grid(row=1, column=0, sticky=tk.W, pady=5)
entry_tmax = ttk.Entry(frame_params2, width=10)
entry_tmax.grid(row=1, column=1, sticky=tk.W, padx=5)
entry_tmax.insert(0, "1")  # Default value

# Number of bootstraps
ttk.Label(frame_params2, text="Iteration").grid(row=2, column=0, sticky=tk.W, pady=5)
entry_nBoot = ttk.Entry(frame_params2, width=10)
entry_nBoot.grid(row=2, column=1, sticky=tk.W, padx=5)
entry_nBoot.insert(0, "100")  # Default value

# Custom suffix for output files
ttk.Label(frame_params2, text="Custom Suffix").grid(row=3, column=0, sticky=tk.W, pady=5)
entry_suffix = ttk.Entry(frame_params2, width=10)
entry_suffix.grid(row=3, column=1, sticky=tk.W, padx=5)
entry_suffix.insert(0, "")
ttk.Label(frame_params2, text="(Opsional)").grid(row=3, column=2, sticky=tk.W)

# Info frame
info_frame = ttk.Frame(left_frame)
info_frame.pack(pady=5, fill=tk.X)

# Info label
info_label = tk.Label(info_frame, text="η₀ akan dihitung otomatis berdasarkan persentil ke-1 dari distribusi NND", 
                      fg="blue", font=("Arial", 9))
info_label.pack(pady=5)

# Run analysis button
run_button = ttk.Button(left_frame, text="Run Clustering Analysis", command=run_analysis)
run_button.pack(pady=10)

# Status label
status_var = tk.StringVar()
status_var.set("Siap menjalankan analisis")
status_label = tk.Label(left_frame, textvariable=status_var, fg="black", font=("Arial", 10, "bold"))
status_label.pack(pady=10)

# Create a notebook for output in the right panel
output_notebook = ttk.Notebook(right_frame)
output_notebook.pack(fill=tk.BOTH, expand=True)

# Add an empty tab as placeholder
empty_tab = ttk.Frame(output_notebook)
output_notebook.add(empty_tab, text="No Results Yet")
tk.Label(empty_tab, text="Load a file and calculate parameters to see results", 
         font=("Arial", 12)).pack(expand=True, pady=50)

root.mainloop()