import os
import numpy as np
import scipy.io
from scipy.stats import gaussian_kde

class SeismicDataProcessor:
    @staticmethod
    def filter_columns(input_file, columns_to_remove):
        """
        Removes specified columns from input file
        """
        temp_file = 'temp_processed.txt'
        filter_commands = []
        for col in columns_to_remove:
            filter_commands.append(f"${col}=\"\"")
            
        command = f"awk '{{ {'; '.join(filter_commands)}; print}}' {input_file} > {temp_file}"
        os.system(command)
        return temp_file

    @staticmethod
    def load_matlab_data(filename, verbose=False):
        """
        Enhanced MATLAB file loader with proper Python dictionary conversion
        """
        def _process_matlab_struct(matobj):
            if isinstance(matobj, scipy.io.matlab.mio5_params.mat_struct):
                return {field: _process_matlab_struct(getattr(matobj, field)) 
                        for field in matobj._fieldnames}
            return matobj

        def _clean_matlab_dict(data_dict):
            return {k: v for k, v in data_dict.items() if not k.startswith('_')}

        raw_data = scipy.io.loadmat(filename, struct_as_record=True, squeeze_me=True)
        processed_data = {k: _process_matlab_struct(v) for k, v in raw_data.items()}
        cleaned_data = _clean_matlab_dict(processed_data)
        
        if verbose:
            print(f"Loaded {len(cleaned_data)} valid entries from {filename}")
            
        return cleaned_data

    @staticmethod
    def calculate_density_2d(x_coords, y_coords, x_bins, y_bins, bandwidth=None):
        """
        Calculates 2D density estimation using Gaussian KDE
        """
        n_points = len(x_coords)
        dims = 2
        
        if bandwidth is None:
            bandwidth = n_points ** (-1.0 / (dims + 2.5))

        kde = gaussian_kde([x_coords, y_coords], bw_method=bandwidth)
        
        X_mesh, Y_mesh = np.meshgrid(x_bins, y_bins)
        positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
        Z = kde(positions).reshape(X_mesh.shape)
        
        dx = x_bins[1] - x_bins[0]
        dy = y_bins[1] - y_bins[0]
        
        integral_check = np.sum(Z) * (dx * dy)
        if abs(integral_check - 1.0) > 0.1:
            print(f"Warning: Density integral = {integral_check:.3f} (should be close to 1.0)")
            
        return (X_mesh - 0.5 * dx, Y_mesh - 0.5 * dy, Z)

    @staticmethod
    def filter_data_range(data_dict, field, min_val=None, max_val=None, include_boundaries=False):
        """
        Filters dictionary data based on value range
        """
        target_array = data_dict[field]
        
        if include_boundaries and (min_val is not None and max_val is not None):
            mask = (target_array >= min_val) & (target_array <= max_val)
        else:
            if min_val is None and max_val is not None:
                mask = target_array < max_val
            elif max_val is None and min_val is not None:
                mask = target_array > min_val
            else:
                mask = (target_array > min_val) & (target_array < max_val)
                
        return {k: v[mask] for k, v in data_dict.items()}