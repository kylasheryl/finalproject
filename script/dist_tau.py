#!/usr/bin/env python3
"""
Modified script for distance-time analysis using new modules
Maintains original GitHub logic
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from src.seismic_data_processor import SeismicDataProcessor
from src.event_grouping import EventGrouping
from src.SeismicCatalog import SeismicCatalog

def setup_catalogs():
    """Initialize all required catalogs"""
    return {
        'original': SeismicCatalog(),
        'mc_filtered': SeismicCatalog(),
        'child_events': SeismicCatalog(),
        'parent_events': SeismicCatalog()
    }

def load_configuration():
    """Load analysis configuration"""
    return {
        'data_dir': 'data',
        'input_file': 'MEQHULULAISALL.mat',
        'magnitude_thresholds': np.array([0.0]),
        'seismic_params': {
            'D': 0.79,
            'b': 0.66
        },
        'plot_params': {
            'bin_size': 0.1,
            'sigma': None,
            'eta_0': -4.810,
            'T_range': (-8, 0),
            'R_range': (-5, 3),
            'colormap': plt.cm.RdYlGn_r
        }
    }

def create_rt_plot(time_data, dist_data, config, eta_0):
    """Create rescaled time-distance plot"""
    data_processor = SeismicDataProcessor()
    
    time_bins = np.arange(
        config['plot_params']['T_range'][0],
        config['plot_params']['T_range'][1] + 2 * config['plot_params']['bin_size'],
        config['plot_params']['bin_size']
    )
    
    dist_bins = np.arange(
        config['plot_params']['R_range'][0],
        config['plot_params']['R_range'][1] + 2 * config['plot_params']['bin_size'],
        config['plot_params']['bin_size']
    )
    
    XX, YY, ZZ = data_processor.calculate_density_2d(
        np.log10(time_data),
        np.log10(dist_data),
        time_bins,
        dist_bins,
        config['plot_params']['sigma']
    )
    
    fig = plt.figure(figsize=(8, 10))
    ax = plt.subplot(111)
    ax.set_title('Nearest Neighbor Pairs in R-T')
    
    norm_ZZ = ZZ * (config['plot_params']['bin_size']**2 * len(time_data))
    plot = ax.pcolormesh(XX, YY, norm_ZZ, cmap=config['plot_params']['colormap'])
    
    cbar = plt.colorbar(plot, orientation='horizontal', shrink=0.5, aspect=20)
    cbar.set_label('Number of Event Pairs', labelpad=-40)
    
    time_range = config['plot_params']['T_range']
    ax.plot(
        [time_range[0], time_range[1]],
        -np.array([time_range[0], time_range[1]]) + eta_0,
        '-',
        lw=1.5,
        color='w'
    )
    ax.plot(
        [time_range[0], time_range[1]],
        -np.array([time_range[0], time_range[1]]) + eta_0,
        '--',
        lw=1.5,
        color='0.5'
    )
    
    ax.set_xlabel('Rescaled Time')
    ax.set_ylabel('Rescaled Distance')
    ax.set_xlim(config['plot_params']['T_range'])
    ax.set_ylim(config['plot_params']['R_range'])
    
    return fig

def save_results(distance, time, config, mc):
    """Save calculated results to file"""
    output_file = os.path.join(
        config['data_dir'],
        f"{config['input_file'].split('.')[0]}_RTall_Mc_{mc:.1f}.mat"
    )
    
    scipy.io.savemat(
        output_file,
        {
            'R': distance,
            'T': time
        },
        do_compression=True
    )
    print(f"Results saved to: {output_file}")

def main():
    # Setup
    catalogs = setup_catalogs()
    config = load_configuration()
    event_grouping = EventGrouping()
    
    try:
        # Load and process initial catalog
        catalogs['original'].load_from_mat(
            os.path.join(config['data_dir'], config['input_file'])
        )
        
        catalogs['original'].filter_by_magnitude(
            config['magnitude_thresholds'][0],
            None
        )
        print(f"Events after initial selection: {catalogs['original'].size()}")
        
        # Convert coordinates
        catalogs['original'].to_cartesian(projection='eqdc')
        
        # Process each magnitude threshold
        for mc in config['magnitude_thresholds']:
            # Load eta_0
            eta_0_file = os.path.join(
                config['data_dir'],
                f"{config['input_file']}_Mc_{mc:.1f}_eta_0all.txt"
            )
            
            if os.path.isfile(eta_0_file):
                eta_0 = np.loadtxt(eta_0_file)
                print(f"Loaded eta_0: {eta_0}")
            else:
                eta_0 = config['plot_params']['eta_0']
                print(f"Using default eta_0: {eta_0}")
            
            # Filter catalog
            catalogs['mc_filtered'].copy_from(catalogs['original'])
            catalogs['mc_filtered'].filter_by_magnitude(mc, None)
            print(f"Processing Mc={mc}, catalog size: {catalogs['mc_filtered'].size()}")
            
            # Load NND results
            nnd_file = os.path.join(
                config['data_dir'],
                f"{config['input_file'].split('.')[0]}_NND_Mcall_{mc:.1f}.mat"
            )
            
            try:
                nnd_data = scipy.io.loadmat(nnd_file)
                print(f"Available keys in NND data: {sorted(nnd_data.keys())}")
                
                # Get IDs and ensure they're 1D arrays
                child_ids = nnd_data['child_ids'].flatten()
                parent_ids = nnd_data['parent_ids'].flatten()
                
                # Create temporary catalogs
                temp_child = SeismicCatalog()
                temp_parent = SeismicCatalog()
                
                # Copy base catalog
                temp_child.copy_from(catalogs['mc_filtered'])
                temp_parent.copy_from(catalogs['mc_filtered'])
                
                # Select events using the modified method name
                temp_child.select_events_by_id(child_ids, allow_repeats=True)
                temp_parent.select_events_by_id(parent_ids, allow_repeats=True)
                
                # Copy to final catalogs
                catalogs['child_events'].copy_from(temp_child)
                catalogs['parent_events'].copy_from(temp_parent)
                
                print(f"Child events: {catalogs['child_events'].size()}, "
                      f"Parent events: {catalogs['parent_events'].size()}")
                
                # Calculate rescaled parameters
                distance, time = event_grouping.calculate_rescaled_parameters(
                    catalogs['child_events'],
                    catalogs['parent_events'],
                    {
                        'b': config['seismic_params']['b'],
                        'D': config['seismic_params']['D'],
                        'Mc': mc
                    }
                )
                
                # Save results and create plots
                save_results(distance, time, config, mc)
                
                fig = create_rt_plot(time, distance, config, eta_0)
                plot_file = os.path.join(
                    'plots',
                    f"T_Rall_{config['input_file'].split('.')[0]}_Mc_{mc:.1f}.png"
                )
                os.makedirs('plots', exist_ok=True)
                fig.savefig(plot_file)
                print(f"Plot saved to: {plot_file}")
                plt.close(fig)
                
            except Exception as e:
                print(f"Error processing NND file: {str(e)}")
                if 'nnd_data' in locals():
                    print("Available keys in NND file:", 
                          sorted(k for k in nnd_data.keys() if not k.startswith('_')))
                raise
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()