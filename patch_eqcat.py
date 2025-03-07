"""
Patch the EqCat class to use fallback cartesian conversion
when Basemap is not available
"""

import sys
import os
import importlib.util
import warnings

def patch_eqcat():
    """
    Patch the EqCat class to use our fallback coordinate conversion if needed
    This function should be called before importing EqCat
    """
    # First check if the src directory exists
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    if not os.path.exists(src_dir):
        return False  # Can't patch what doesn't exist
        
    eqcat_path = os.path.join(src_dir, 'EqCat.py')
    if not os.path.exists(eqcat_path):
        return False  # EqCat.py doesn't exist
    
    # Load the EqCat module
    spec = importlib.util.spec_from_file_location("EqCat", eqcat_path)
    eqcat_module = importlib.util.module_from_spec(spec)
    sys.modules["EqCat"] = eqcat_module
    spec.loader.exec_module(eqcat_module)
    
    # Check if we need to patch it
    try:
        import mpl_toolkits.basemap
        # Basemap exists, no need to patch
        return True
    except ImportError:
        # Basemap not available, patch the method
        from coordinate_conversion import simple_cartesian_conversion
        
        # Store the original method
        original_method = eqcat_module.EqCat.toCart_coordinates
        
        # Create a patched method that uses our simple conversion
        def patched_toCart_coordinates(self, projection='eqdc'):
            warnings.warn("Basemap not available, using simple cartesian conversion instead")
            try:
                return simple_cartesian_conversion(self)
            except Exception as e:
                print(f"Error in coordinate conversion: {e}")
                raise
        
        # Replace the method
        eqcat_module.EqCat.toCart_coordinates = patched_toCart_coordinates
        print("Successfully patched EqCat.toCart_coordinates to work without Basemap")
        return True

if __name__ == "__main__":
    # Can be run directly to patch EqCat
    result = patch_eqcat()
    if result:
        print("Successfully patched EqCat")
    else:
        print("Failed to patch EqCat")
