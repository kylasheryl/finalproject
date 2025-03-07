"""
Custom coordinate conversion functions for earthquake analysis
Provides a fallback method when Basemap is not available
"""

import numpy as np
import warnings

def simple_cartesian_conversion(eqCat):
    """
    Simple cartesian coordinate conversion without requiring Basemap
    Uses an equirectangular projection centered on the data
    
    Parameters:
        eqCat: EqCat object with Lat and Lon data
    
    Returns:
        None (modifies eqCat in place, adding 'X' and 'Y' keys)
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Get reference point (center of the data)
    lat0 = np.mean(eqCat.data['Lat'])
    lon0 = np.mean(eqCat.data['Lon'])
    
    # Convert to radians
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat_rad = np.radians(eqCat.data['Lat'])
    lon_rad = np.radians(eqCat.data['Lon'])
    
    # Calculate X and Y coordinates (equirectangular projection)
    # X = R * (lon - lon0) * cos(lat0)
    # Y = R * (lat - lat0)
    eqCat.data['X'] = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    eqCat.data['Y'] = R * (lat_rad - lat0_rad)
    
    return

def try_install_basemap():
    """
    Attempt to install Basemap using pip
    Returns True if successful, False otherwise
    """
    try:
        import importlib
        
        # Check if basemap is already installed
        try:
            importlib.import_module('mpl_toolkits.basemap')
            print("Basemap is already installed.")
            return True
        except ImportError:
            pass
            
        # Try to install basemap
        import subprocess
        import sys
        
        print("Attempting to install Basemap...")
        result = subprocess.call([sys.executable, "-m", "pip", "install", "basemap"])
        
        if result == 0:
            print("Basemap installed successfully!")
            return True
        else:
            print("Failed to install Basemap via pip.")
            return False
            
    except Exception as e:
        print(f"Error trying to install Basemap: {str(e)}")
        return False

def safe_cartesian_conversion(eqCat, projection='eqdc'):
    """
    Safely convert to cartesian coordinates, falling back to a simple method if Basemap fails
    
    Parameters:
        eqCat: EqCat object with Lat and Lon data
        projection: Projection to use if Basemap is available
    
    Returns:
        None (modifies eqCat in place, adding 'X' and 'Y' keys)
    """
    # Try to use the original toCart_coordinates method
    try:
        original_result = eqCat.toCart_coordinates(projection=projection)
        return original_result
    except Exception as basemap_error:
        warnings.warn(f"Basemap conversion failed: {str(basemap_error)}. Using simple cartesian conversion instead.")
        
        # If that fails, use our simple cartesian conversion
        simple_cartesian_conversion(eqCat)
        return
