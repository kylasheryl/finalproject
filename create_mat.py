import os
import sys
import numpy as np

# Add the parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from src.SeismicCatalog import SeismicCatalog

def main():
    # Initialize catalog
    seismic_catalog = SeismicCatalog()

    # Set input/output paths
    data_dir = 'data'
    input_file = 'MEQHULULAISALL.txt'
    output_file = os.path.join(data_dir, input_file.replace('txt', 'mat'))

    # Ensure output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load data
    try:
        seismic_catalog.load_catalog(
            os.path.join(data_dir, input_file), 
            'HS_reloc'
        )
        print(f'Total events loaded: {seismic_catalog.size()}')
        print(f'Available data fields: {sorted(seismic_catalog.data.keys())}')

        # Save to MAT file
        seismic_catalog.save_to_mat(output_file)
        print(f"Data saved to: {output_file}")

        # Verify saved data
        verification_catalog = SeismicCatalog()
        verification_catalog.load_from_mat(output_file)
        print(f"Verification size: {verification_catalog.size()}")
        print(f"Verification fields: {sorted(verification_catalog.data.keys())}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
