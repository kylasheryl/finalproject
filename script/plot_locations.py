#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from src.SeismicCatalog import SeismicCatalog

def setup_basemap():
    """Setup basemap with proper environment variables"""
    try:
        if sys.platform == 'win32':
            import site
            site_packages = site.getsitepackages()[0]
            proj_path = os.path.join(os.path.dirname(site_packages), 'Library', 'share', 'proj')
            if os.path.exists(proj_path):
                os.environ["PROJ_LIB"] = proj_path
        else:
            os.environ["PROJ_LIB"] = f"{os.environ.get('HOME', '')}/opt/anaconda3/share/proj"
            
        from mpl_toolkits.basemap import Basemap
        return Basemap
    except ImportError:
        print("Error: Basemap not installed. Install with: pip install basemap")
        sys.exit(1)

def main():
    # Configuration
    input_file = "MEQHULULAISALL.mat"
    mag_range = (0, None)  # Mmin, Mmax
    time_range = (2020, 2022)  # tmin, tmax
    
    try:
        # Load data
        catalog = SeismicCatalog()
        catalog.load_from_mat(input_file)
        
        # Apply filters
        catalog.filter_by_magnitude(*mag_range)
        catalog.filter_by_time(*time_range)
        
        if catalog.size() == 0:
            print("Error: No data meets selection criteria")
            sys.exit(1)
            
        # Setup map
        Basemap = setup_basemap()
        margin = 0.5
        
        bounds = {
            'xmin': catalog.data['Lon'].min() - margin,
            'xmax': catalog.data['Lon'].max() + margin,
            'ymin': catalog.data['Lat'].min() - margin,
            'ymax': catalog.data['Lat'].max() + margin
        }
        
        # Create map
        m = Basemap(
            llcrnrlat=bounds['ymin'], urcrnrlat=bounds['ymax'],
            llcrnrlon=bounds['xmin'], urcrnrlon=bounds['xmax'],
            projection='cyl', resolution='l'
        )
        
        # Draw map elements
        m.drawstates(linewidth=1)
        m.drawcoastlines(linewidth=2)
        
        # Plot earthquakes
        x, y = m(catalog.data['Lon'], catalog.data['Lat'])
        m.plot(x, y, 'ko', ms=1, label='Earthquakes')
        
        # Highlight larger earthquakes
        large_eq_mask = catalog.data['Mag'] >= 1.9
        m.plot(x[large_eq_mask], y[large_eq_mask], 'ro', ms=8, 
               mew=1.5, mfc='none', label='M >= 2.0')
        
        # Draw grid
        m.drawmeridians(
            np.linspace(int(bounds['xmin']), bounds['xmax'], 4),
            labels=[False, False, False, True],
            fontsize=10
        )
        m.drawparallels(
            np.linspace(int(bounds['ymin']), bounds['ymax'], 4),
            labels=[True, False, False, False],
            fontsize=10
        )
        
        # Finalize plot
        plt.legend()
        plt.title("Regional Area")
        plt.savefig("earthquake_catalog_regional.png")
        plt.show()
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()