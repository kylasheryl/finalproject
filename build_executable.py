"""
Build Executable for Earthquake Clustering Analysis GUI
-------------------------------------------------------
This script creates a Windows executable (.exe) file using PyInstaller.
It completely avoids the Basemap dependency.

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

# Make sure main script exists
SOURCE_FILE = "earthquake_clustering_gui.py"
if not os.path.exists(SOURCE_FILE):
    print(f"Error: Could not find source file '{SOURCE_FILE}'")
    print("Please make sure your main script is named 'earthquake_clustering_gui.py'")
    sys.exit(1)

# Check if the script needs patching
print("Checking if the GUI script needs patching...")
with open(SOURCE_FILE, 'r') as f:
    gui_content = f.read()

# Define the patch code
patch_code = """
# ------ BASEMAP AVOIDANCE PATCH ------
import os
import sys
import warnings

# Before any other imports, disable Basemap completely
# This prevents even the import attempt that's causing the error
sys.modules['mpl_toolkits.basemap'] = None

# Define our own cartesian conversion function
def simple_cartesian_conversion(eqCat):
    \"\"\"Simple cartesian coordinate conversion without Basemap\"\"\"
    import numpy as np
    
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
    eqCat.data['X'] = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    eqCat.data['Y'] = R * (lat_rad - lat0_rad)
    
    return

# Enable this flag for your functions to use
use_simple_conversion = True
# ------ END OF BASEMAP AVOIDANCE PATCH ------
"""

# Check if the patch is already applied
if "sys.modules['mpl_toolkits.basemap'] = None" not in gui_content:
    print("Patching the GUI script to completely avoid Basemap...")
    
    # Find the first import statement
    import_lines = [i for i, line in enumerate(gui_content.split('\n')) if 
                   line.strip().startswith('import ') or line.strip().startswith('from ')]
    
    if import_lines:
        first_import = import_lines[0]
        # Insert our patch before the first import
        lines = gui_content.split('\n')
        patched_content = '\n'.join(lines[:first_import]) + patch_code + '\n'.join(lines[first_import:])
        
        # Now find the cartesian conversion calls and replace them
        patched_content = patched_content.replace(
            "eqCat.toCart_coordinates(projection='eqdc')", 
            "simple_cartesian_conversion(eqCat)"
        )
        
        # Save the patched file
        with open(SOURCE_FILE, 'w') as f:
            f.writelines(patched_content)
        
        print("GUI script patched successfully to avoid Basemap completely")
    else:
        print("Warning: Could not find import statements in the script. Manual patching may be required.")
else:
    print("GUI script already patched to avoid Basemap")

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

# Create a runtime hook to exclude Basemap
hook_file = "disable_basemap_hook.py"
with open(hook_file, "w") as f:
    f.write("""
# Runtime hook to disable Basemap import
import sys
import warnings

# Disable the import warning
warnings.filterwarnings('ignore', message='.*basemap.*')

# Mock the Basemap module
sys.modules['mpl_toolkits.basemap'] = None
""")
print(f"Created runtime hook to disable Basemap")

# Build the PyInstaller command
command = [
    "pyinstaller",
    "--name", APP_NAME,
    "--onefile",
    "--windowed",
    "--runtime-hook", hook_file,
    # Exclude problematic packages
    "--exclude-module", "mpl_toolkits.basemap",
    "--exclude-module", "basemap",
]

# Add icon if available
if ICON_FILE and os.path.exists(ICON_FILE):
    command.extend(["--icon", ICON_FILE])

# Add data folders if found
for folder in data_folders:
    command.extend(["--add-data", f"{folder};{folder}"])

# Add hidden imports for difficult modules
command.extend(["--hidden-import", "scipy.spatial.transform._rotation_groups"])

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
                
    else:
        print("\nError: Could not find the generated executable.")
        print("Check the 'dist' directory manually.")
except subprocess.CalledProcessError as e:
    print(f"\nError building executable: {e}")
    print("See PyInstaller output above for details.")

print("\nNote: This executable completely avoids using Basemap.")
print("It uses a simplified coordinate conversion method instead.")
print("\nTo distribute your application, share the contents of the 'dist' folder.")