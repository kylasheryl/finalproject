import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from src.SeismicCatalog import SeismicCatalog
from src.event_grouping import EventGrouping
from src.seismic_data_processor import SeismicDataProcessor

def main():
    # Initialize catalogs
    original_catalog = SeismicCatalog()
    random_catalog = SeismicCatalog()
    mc_catalog = SeismicCatalog()
    
    np.random.seed(123456)
    
    # Configuration
    config = {
        'data_dir': 'data',
        'input_file': 'MEQHULULAISALL.mat',
        'mc_values': np.array([0.0]),
        'seismic_params': {
            'D': 0.79,
            'b': 0.66
        },
        'bootstrap_count': 100,
        'plot_params': {
            'eta_binsize': 0.3,
            'cmin': 1,
            'xmin': -13, 'xmax': 0,
            'T_range': (-8, 0),
            'R_range': (-5, 3),
            'bin_size': 0.1,
            'colormap': plt.cm.RdYlGn_r
        }
    }
    
    # Load and process data
    try:
        original_catalog.load_from_mat(
            os.path.join(config['data_dir'], config['input_file'])
        )
        print(f"Total events: {original_catalog.size()}")
        
        # Initial magnitude filter
        original_catalog.filter_by_magnitude(config['mc_values'][0], None)
        print(f"Events after initial selection: {original_catalog.size()}")
        
        # Convert to cartesian coordinates
        original_catalog.to_cartesian(projection='eqdc')
        
        event_grouping = EventGrouping()
        
        for mc in config['mc_values']:
            print(f"\nProcessing Mc = {mc}")
            
            # Copy and filter catalog
            mc_catalog.copy_from(original_catalog)
            mc_catalog.filter_by_magnitude(mc, None)
            
            # Calculate eta_0 through bootstrapping
            eta_0_values = np.zeros(config['bootstrap_count'])
            
            for i in range(config['bootstrap_count']):
                # Create randomized catalog
                random_catalog.copy_from(mc_catalog)
                random_catalog.randomize_locations_and_times()
                
                # Calculate NND
                nnd_results = event_grouping.find_nearest_neighbors(
                    random_catalog,
                    {'Mc': mc, 'b': config['seismic_params']['b'], 
                     'D': config['seismic_params']['D']}
                )
                
                eta_0_values[i] = np.percentile(
                    np.log10(nnd_results['distances']), 1
                )
                print(f"Bootstrap {i+1}/{config['bootstrap_count']}, "
                      f"eta_0 = {eta_0_values[i]:.5f}")
                
            # Save results
            mean_eta_0 = eta_0_values.mean()
            print(f"Mean eta_0: {mean_eta_0}")
            
            output_file = os.path.join(
                config['data_dir'],
                f"{config['input_file']}_Mc_{mc:.1f}_eta_0all.txt"
            )
            
            np.savetxt(
                output_file,
                np.array([mean_eta_0]),
                fmt='%10.3f',
                header='eta_0'
            )
            
            # Save MAT file
            scipy.io.savemat(
                output_file.replace('txt', 'mat'),
                {
                    'eta_0': mean_eta_0,
                    'eta_BS': eta_0_values
                },
                do_compression=True
            )
            print(f"Results saved to {output_file}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
