#!/usr/bin/env python3
'''
Modified version for nearest-neighbor distance analysis 
Maintains original GitHub workflow
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from src.SeismicCatalog import SeismicCatalog
from src.event_grouping import EventGrouping

def main():
    # Initialize catalog
    catalog = SeismicCatalog()
    event_grouping = EventGrouping()

    # Configuration - using original parameters
    config = {
        'data_dir': 'data',
        'input_file': 'MEQHULULAISALL.mat',
        'mc_values': np.array([0.0]),
        'seismic_params': {
            'D': 0.79,  # fractal dimension for eq. (1)
            'b': 0.66,  # use: https://github.com/tgoebel/magnitude-distribution for b-value
        },
        'eta_binsize': 0.3,
        'xmin': -13, 
        'xmax': 0
    }

    try:
        # Load catalog
        catalog.load_from_mat(os.path.join(config['data_dir'], config['input_file']))
        print('total no. of events', catalog.size())

        # Initial magnitude filter
        catalog.filter_by_magnitude(config['mc_values'][0], None)
        print('no. of events after initial selection', catalog.size())

        # Convert to cartesian coordinates
        catalog.to_cartesian(projection='eqdc')

        # Process each magnitude threshold
        for mc in config['mc_values']:
            print('-------------- current Mc:', mc, '---------------------')

            # Magnitude selection
            catalog.filter_by_magnitude(mc, None)
            print('catalog size after MAG selection', catalog.size())

            # Parameters for NND calculation
            params = {
                'Mc': mc,
                'b': config['seismic_params']['b'],
                'D': config['seismic_params']['D']
            }

            # Calculate NND
            print('depth range: ', catalog.data['Depth'].min(), catalog.data['Depth'].max())
            nnd_results = event_grouping.find_nearest_neighbors(
                catalog, 
                params,
                distance_3D=False,
                correct_colocated=True,
                verbose=True
            )

            # Create histogram
            bins = np.arange(-13, 1, config['eta_binsize'], dtype=float)
            hist, bin_edges = np.histogram(
                np.log10(nnd_results['distances'][nnd_results['distances'] > 0]), 
                bins
            )
            bin_centers = bin_edges[:-1] + config['eta_binsize'] * 0.5

            # Convert to PDF
            hist = hist / config['eta_binsize']  # Correct for binsize
            hist = hist / catalog.size()  # Convert to PDF

            # Save NND results
            nnd_file = f"data/{config['input_file'].split('.')[0]}_NND_Mcall_{mc:.1f}.mat"
            print('save file', nnd_file)
            scipy.io.savemat(nnd_file, nnd_results, do_compression=True)

            # Load eta_0 value for plotting
            eta_0_file = f"{config['data_dir']}/{config['input_file']}_Mc_{mc:.1f}_eta_0all.txt"
            if os.path.isfile(eta_0_file):
                print('load eta_0 from file')
                eta_0 = np.loadtxt(eta_0_file)
                print('eta_0', eta_0)
            else:
                eta_0 = -5
                print('could not find eta_0 file', eta_0_file, 'use value: ', eta_0)

            # Create plot
            fig, ax = plt.subplots()
            ax.bar(
                bin_centers, 
                hist, 
                width=0.8 * config['eta_binsize'],
                align='edge',
                color='.5',
                label=f'Mc = {mc:.1f}'
            )

            # Add reference lines
            ax.plot(
                [eta_0, eta_0], 
                ax.get_ylim(), 
                'w-', 
                lw=2,
                label=f'N_total={catalog.size()}'
            )
            ax.plot(
                [eta_0, eta_0], 
                ax.get_ylim(), 
                'r--', 
                lw=2,
                label=f'N_cl={np.sum(nnd_results["distances"] < 1e-5)}'
            )

            ax.legend(loc='upper left')
            ax.set_xlabel('NND, log$_{10} \\eta$')
            ax.set_ylabel('Number of Events')
            ax.grid(True)
            ax.set_xlim(config['xmin'], config['xmax'])

            # Save plot
            plot_file = f'plots/{config["input_file"].split(".")[0]}_NND_hist_Mcall_{mc:.1f}.png'
            os.makedirs('plots', exist_ok=True)
            print('save plot', plot_file)
            fig.savefig(plot_file)
            plt.close(fig)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()