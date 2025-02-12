import sys
import os
import numpy as np
import scipy.io

# Fix the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from time_converter import TimeConverter  # Jika di dalam folder src

class SeismicCatalog:
    """
    Enhanced seismic catalog class with modern Python practices
    """
    def __init__(self):
        self.data = {}
        self.location_fields = ('Lon', 'Lat', 'Depth')
        self.id_field = 'N'

class SeismicCatalog:
    """
    Enhanced seismic catalog class with modern Python practices
    """
    def __init__(self):
        self.data = {}
        self.location_fields = ('Lon', 'Lat', 'Depth')
        self.id_field = 'N'
        
    def copy_from(self, other_catalog):
        """Deep copy from another catalog"""
        import copy
        if isinstance(other_catalog, dict):
            self.data = {k: copy.deepcopy(v) for k, v in other_catalog.items()}
        else:
            self.data = {k: copy.deepcopy(v) for k, v in other_catalog.data.items()}

    def size(self):
        """Get number of events in catalog"""
        return len(self.data.get('Time', [])) if 'Time' in self.data else 0

    def load_catalog(self, filename, catalog_type, verbose=False, **kwargs):
        """
        Load earthquake catalog from file
        """
        if catalog_type == 'HS_reloc':
            try:
                # Load data columns
                data_matrix = np.loadtxt(filename, usecols=(0,1,2,3,4,5,6,7,8,9,10))
                
                # Map columns to data dictionary
                columns = ['YR', 'MO', 'DY', 'HR', 'MN', 'SC', 'Lat', 'Lon', 'Depth', 'Mag', 'N']
                for i, col in enumerate(columns):
                    self.data[col] = data_matrix[:, i]
                
                # Convert time to decimal years
                self.data['Time'] = np.array([
                    TimeConverter.datetime_to_decimal_year(
                        self.data['YR'][i], self.data['MO'][i],
                        self.data['DY'][i], self.data['HR'][i],
                        self.data['MN'][i], self.data['SC'][i]
                    ) for i in range(len(self.data['YR']))
                ])
                
                # Sort chronologically
                self.sort_catalog('Time')
                
            except Exception as e:
                raise ValueError(f"Error loading catalog: {str(e)}")

    def filter_by_magnitude(self, min_mag, max_mag=None, include_bounds=False):
        """Filter events by magnitude range"""
        if include_bounds:
            if min_mag is not None and max_mag is not None:
                mask = (self.data['Mag'] >= min_mag) & (self.data['Mag'] <= max_mag)
        else:
            if max_mag is None:
                mask = self.data['Mag'] >= min_mag
            elif min_mag is None:
                mask = self.data['Mag'] < max_mag
            else:
                mask = (self.data['Mag'] >= min_mag) & (self.data['Mag'] < max_mag)
                
        self.apply_filter(mask)

    def filter_by_time(self, start_time, end_time, include_bounds=False):
        """Filter events by time range"""
        if include_bounds:
            mask = (self.data['Time'] >= start_time) & (self.data['Time'] <= end_time)
        else:
            mask = (self.data['Time'] >= start_time) & (self.data['Time'] < end_time)
        self.apply_filter(mask)

    def apply_filter(self, mask):
        """Apply boolean mask to all data fields"""
        for key in self.data:
            self.data[key] = self.data[key][mask]

    def sort_catalog(self, field, descending=False):
        """Sort catalog by specified field"""
        sort_idx = np.argsort(self.data[field])
        if descending:
            sort_idx = sort_idx[::-1]
        
        for key in self.data:
            self.data[key] = self.data[key][sort_idx]

    def to_cartesian(self, projection='aeqd'):
        """Convert coordinates to cartesian system"""
        try:
            import site
            if sys.platform == 'win32':
                site_packages = site.getsitepackages()[0]
                proj_path = os.path.join(os.path.dirname(site_packages), 'Library', 'share', 'proj')
                if os.path.exists(proj_path):
                    os.environ["PROJ_LIB"] = proj_path
            
            from mpl_toolkits.basemap import Basemap
            
            bounds = {
                'xmin': self.data['Lon'].min(),
                'xmax': self.data['Lon'].max(),
                'ymin': self.data['Lat'].min(),
                'ymax': self.data['Lat'].max()
            }
            
            m = Basemap(
                llcrnrlat=bounds['ymin'], urcrnrlat=bounds['ymax'],
                llcrnrlon=bounds['xmin'], urcrnrlon=bounds['xmax'],
                projection=projection,
                lat_0=(bounds['ymin'] + bounds['ymax']) * 0.5,
                lon_0=(bounds['xmin'] + bounds['xmax']) * 0.5,
                resolution='l'
            )
            
            x, y = m(self.data['Lon'], self.data['Lat'])
            
            if projection != 'cyl':
                x *= 1e-3
                y *= 1e-3
                
            self.data['X'] = x
            self.data['Y'] = y
            
            return True
            
        except ImportError:
            print("Warning: Basemap not available. Install with: pip install basemap")
            return False
        except Exception as e:
            print(f"Error in coordinate conversion: {str(e)}")
            return False

    def save_to_mat(self, filename):
        """Save catalog to MATLAB format file"""
        scipy.io.savemat(
            filename,
            self.data,
            appendmat=True,
            format='5',
            do_compression=True
        )

    def load_from_mat(self, filename):
        """Load catalog from MATLAB format file"""
        def process_matlab_struct(matobj):
            if isinstance(matobj, scipy.io.matlab.mio5_params.mat_struct):
                return {field: process_matlab_struct(getattr(matobj, field))
                        for field in matobj._fieldnames}
            return matobj

        raw_data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        self.data = {k: v for k, v in raw_data.items() if not k.startswith('_')}
        
        for key in list(self.data.keys()):
            if isinstance(self.data[key], scipy.io.matlab.mio5_params.mat_struct):
                self.data[key] = process_matlab_struct(self.data[key])

    def randomize_locations_and_times(self):
        """Create randomized catalog maintaining same statistics"""
        if 'X' in self.data and 'Y' in self.data:
            self.data['X'] = np.random.uniform(
                self.data['X'].min(),
                self.data['X'].max(),
                size=self.size()
            )
            self.data['Y'] = np.random.uniform(
                self.data['Y'].min(),
                self.data['Y'].max(),
                size=self.size()
            )
        
        time_min = self.data['Time'].min()
        time_max = self.data['Time'].max()
        self.data['Time'] = np.random.uniform(time_min, time_max, size=self.size())
        self.sort_catalog('Time')
    def select_events_by_id(self, event_ids, allow_repeats=False):
        """
        Modern version of event selection with better error handling
        Parameters:
        -----------
        event_ids : array-like
            List of event IDs to select
        allow_repeats : bool, optional
            If True, keep repeated events in the order they appear
            If False, keep only first occurrence of each event
        """
        if allow_repeats:
            selection = np.zeros(len(event_ids), dtype=int)
            indices = np.arange(self.size())
            
            for i, event_id in enumerate(event_ids):
                matches = self.data['N'] == int(event_id)
                if np.any(matches):
                    selection[i] = indices[matches][0]
                else:
                    raise ValueError(f"Event ID {event_id} not found in catalog")
        else:
            selection = np.in1d(self.data['N'], event_ids, assume_unique=True)
            selection = np.where(selection)[0]
            
        self.apply_filter(selection)