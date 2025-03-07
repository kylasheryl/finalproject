"""
Build Executable for Earthquake Clustering Analysis GUI
-------------------------------------------------------
This script creates a Windows executable (.exe) file using PyInstaller.

Usage:
1. Save your main GUI script as 'earthquake_clustering_gui.py'
2. Run this script: python build_executable.py
3. The executable will be created in the 'dist' folder

Requirements:
- PyInstaller
- All dependencies used in the main script
"""

import os
import sys
import subprocess
import platform
import shutil
import importlib

# Define the name for your application
APP_NAME = "Earthquake_Clustering_Analysis"

# Check if we're on Windows
if platform.system() != "Windows":
    print("This script is designed to create Windows executables.")
    print("Please run it on a Windows system.")
    sys.exit(1)

def install_package(package):
    """Install a package using pip if not already installed"""
    try:
        importlib.import_module(package)
        print(f"{package} is already installed.")
        return True
    except ImportError:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to install {package} using pip.")
            return False

# Install PyInstaller if not already installed
install_package("pyinstaller")

# Try to install basemap (which can be problematic)
try:
    # Check if basemap is already installed
    try:
        import mpl_toolkits.basemap
        print("Basemap is already installed.")
        basemap_installed = True
    except ImportError:
        print("Basemap is not installed. Attempting to install...")
        
        # First try installing via pip
        result = subprocess.call([sys.executable, "-m", "pip", "install", "basemap"])
        basemap_installed = (result == 0)
        
        if not basemap_installed:
            print("Warning: Could not install Basemap via pip.")
            print("You may need to install it with conda: conda install basemap")
            print("Continuing with build process, but the application might use a fallback method for coordinate conversion.")
except Exception as e:
    print(f"Error checking/installing Basemap: {e}")
    basemap_installed = False

# Create a coordinate conversion module if basemap isn't installed
if not basemap_installed:
    print("Creating a fallback cartesian conversion module...")
    # Create coordinate_conversion.py with fallback methods
    with open("coordinate_conversion.py", "w") as f:
        f.write("""
import numpy as np
import warnings

def simple_cartesian_conversion(eqCat):
    \"\"\"
    Simple cartesian coordinate conversion without requiring Basemap
    Uses an equirectangular projection centered on the data
    
    Parameters:
        eqCat: EqCat object with Lat and Lon data
    
    Returns:
        None (modifies eqCat in place, adding 'X' and 'Y' keys)
    \"\"\"
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

def safe_cartesian_conversion(eqCat, projection='eqdc'):
    \"\"\"
    Safely convert to cartesian coordinates, falling back to a simple method if Basemap fails
    
    Parameters:
        eqCat: EqCat object with Lat and Lon data
        projection: Projection to use if Basemap is available
    
    Returns:
        None (modifies eqCat in place, adding 'X' and 'Y' keys)
    \"\"\"
    # Try to use the original toCart_coordinates method
    try:
        original_result = eqCat.toCart_coordinates(projection=projection)
        return original_result
    except Exception as basemap_error:
        warnings.warn(f"Basemap conversion failed: {str(basemap_error)}. Using simple cartesian conversion instead.")
        
        # If that fails, use our simple cartesian conversion
        simple_cartesian_conversion(eqCat)
        return
""")
    print("Created coordinate_conversion.py with fallback methods")

# Check for the patch_eqcat script and create it if needed
if not os.path.exists("patch_eqcat.py"):
    print("Creating a script to patch the EqCat class...")
    with open("patch_eqcat.py", "w") as f:
        f.write("""
# Patch the EqCat class to use the fallback cartesian conversion when needed
import sys
import os
import importlib.util

def patch_eqcat():
    \"\"\"
    Patch the EqCat class to use our fallback coordinate conversion if needed
    This function should be called before importing EqCat
    \"\"\"
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
        from coordinate_conversion import safe_cartesian_conversion
        
        # Store the original method
        original_method = eqcat_module.EqCat.toCart_coordinates
        
        # Create a patched method that uses our safe conversion
        def patched_toCart_coordinates(self, projection='eqdc'):
            try:
                return safe_cartesian_conversion(self, projection)
            except Exception as e:
                print(f"Error in coordinate conversion: {e}")
                raise
        
        # Replace the method
        eqcat_module.EqCat.toCart_coordinates = patched_toCart_coordinates
        return True
""")
    print("Created patch_eqcat.py to handle Basemap issues")

# Make sure main script exists
SOURCE_FILE = "earthquake_clustering_gui.py"
if not os.path.exists(SOURCE_FILE):
    print(f"Error: Could not find source file '{SOURCE_FILE}'")
    print("Please make sure your main script is named 'earthquake_clustering_gui.py'")
    sys.exit(1)

# Check if we need to patch the GUI script
patch_needed = not basemap_installed
if patch_needed:
    print("Checking if the GUI script needs patching for coordinate conversion...")
    with open(SOURCE_FILE, 'r') as f:
        gui_content = f.read()
    
    # Check if the script already includes our patch
    if "from coordinate_conversion import" not in gui_content and "import patch_eqcat" not in gui_content:
        print("Patching the GUI script to handle missing Basemap...")
        # Add import for the patch at the beginning of the file, after other imports
        with open(SOURCE_FILE, 'r') as f:
            lines = f.readlines()
        
        # Find a good spot to insert our patch - after imports but before code
        insert_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                insert_line = i + 1
        
        # Add our patch code
        patch_code = [
            "\n# Handle missing Basemap library if needed\n",
            "try:\n",
            "    # Try to import patch_eqcat if it exists\n",
            "    import patch_eqcat\n",
            "    patch_eqcat.patch_eqcat()\n",
            "except ImportError:\n",
            "    # If not available, try direct import of coordinate_conversion\n",
            "    try:\n",
            "        from coordinate_conversion import safe_cartesian_conversion\n",
            "    except ImportError:\n",
            "        pass  # Will use original method, hoping Basemap is available\n\n"
        ]
        
        lines = lines[:insert_line] + patch_code + lines[insert_line:]
        
        # Also patch the calculate_parameters and run_analysis functions to use safe conversion
        for i, line in enumerate(lines):
            if "eqCat.toCart_coordinates(projection='eqdc')" in line:
                indent = ' ' * (len(line) - len(line.lstrip()))
                lines[i] = line.replace(
                    "eqCat.toCart_coordinates(projection='eqdc')", 
                    "safe_cartesian_conversion(eqCat, projection='eqdc') if 'safe_cartesian_conversion' in globals() else eqCat.toCart_coordinates(projection='eqdc')"
                )
        
        # Save the patched file
        with open(SOURCE_FILE, 'w') as f:
            f.writelines(lines)
        
        print("GUI script patched successfully to handle missing Basemap")

# Create a folder for the icon if it doesn't exist
if not os.path.exists("icons"):
    os.makedirs("icons")
    print("Created 'icons' directory")

# Create a simple icon file if none exists
ICON_FILE = "icons/eq_icon.ico"
if not os.path.exists(ICON_FILE):
    print("No icon file found. Creating a default one...")
    # Try to generate a simple icon (requires Pillow)
    try:
        from PIL import Image, ImageDraw
        
        # Create a 256x256 black icon with a white seismograph line
        img = Image.new('RGB', (256, 256), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple seismograph line
        center_y = 128
        amplitude = 40
        x_positions = range(40, 216, 4)
        y_positions = [center_y + amplitude * (0.5 if i % 12 == 0 else 
                                             -0.8 if i % 10 == 0 else 
                                             0.3 if i % 7 == 0 else 
                                             -0.4 if i % 5 == 0 else 0.1) 
                      for i in range(len(x_positions))]
        
        points = list(zip(x_positions, y_positions))
        draw.line(points, fill=(255, 255, 255), width=3)
        
        # Save as PNG first
        png_path = ICON_FILE.replace('.ico', '.png')
        img.save(png_path)
        
        # Convert PNG to ICO (requires Pillow)
        img = img.resize((256, 256))
        img.save(ICON_FILE)
        print(f"Created default icon at {ICON_FILE}")
    except Exception as e:
        print(f"Could not create icon: {e}")
        ICON_FILE = None

# Install other required packages
required_packages = ["numpy", "scipy", "matplotlib", "pandas"]
for package in required_packages:
    install_package(package)

# Check if we need to include additional data files
print("Checking for additional data files...")
data_folders = []
for folder in ["src", "data"]:
    if os.path.exists(folder) and os.path.isdir(folder):
        data_folders.append(folder)
        print(f"Found data folder: {folder}")

# Add the patching scripts to the data files if they exist
extra_data_files = []
for file in ["coordinate_conversion.py", "patch_eqcat.py"]:
    if os.path.exists(file):
        extra_data_files.append((file, "."))

# Build the PyInstaller command
command = [
    "pyinstaller",
    "--name", APP_NAME,
    "--onefile",
    "--windowed",
]

# Add icon if available
if ICON_FILE and os.path.exists(ICON_FILE):
    command.extend(["--icon", ICON_FILE])

# Add data folders if found
for folder in data_folders:
    command.extend(["--add-data", f"{folder};{folder}"])

# Add the additional data files if any
for src, dst in extra_data_files:
    command.extend(["--add-data", f"{src};{dst}"])

# Add hidden imports for difficult modules
command.extend(["--hidden-import", "scipy.spatial.transform._rotation_groups"])
command.extend(["--hidden-import", "mpl_toolkits.basemap"]) # Try to include basemap even if not available

# Add the source file
command.append(SOURCE_FILE)

# Print the command
print("\nExecuting command:")
print(" ".join(command))
print("\nBuilding executable... This may take a few minutes.")

# Run PyInstaller
try:
    subprocess.check_call(command)
    
    # Check if the executable was created
    exe_path = os.path.join("dist", f"{APP_NAME}.exe")
    if os.path.exists(exe_path):
        print("\n" + "="*60)
        print(f"Success! Executable created at: {exe_path}")
        print("="*60)
        
        # Copy a sample CSV file if it exists to help users get started
        for ext in ['.csv', '.mat']:
            sample_files = [f for f in os.listdir('.') if f.endswith(ext)]
            if sample_files:
                sample_dir = os.path.join("dist", "sample_data")
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                for sample in sample_files[:3]:  # Copy up to 3 sample files
                    shutil.copy(sample, os.path.join(sample_dir, sample))
                print(f"Copied sample {ext} files to {sample_dir}")
                
        # Copy patch files to the dist directory if they exist
        for file in ["coordinate_conversion.py", "patch_eqcat.py"]:
            if os.path.exists(file):
                shutil.copy(file, os.path.join("dist", file))
                
    else:
        print("\nError: Could not find the generated executable.")
        print("Check the 'dist' directory manually.")
except subprocess.CalledProcessError as e:
    print(f"\nError building executable: {e}")
    print("See PyInstaller output above for details.")

print("\nNote: If your application requires additional data files or modules,")
print("you may need to modify this script to include them.")
print("\nTo distribute your application, share the contents of the 'dist' folder.")
