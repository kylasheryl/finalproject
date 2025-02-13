import numpy as np
from typing import Dict, Any, Tuple

class EventGrouping:
    """
    Enhanced class for earthquake event clustering and analysis
    """
    
    def __init__(self):
        """Initialize event grouping processor"""
        self.EARTH_RADIUS = 6371  # km

    def calculate_haversine_distance(self, lon1: float, lat1: float, 
                                   lon2: float, lat2: float) -> float:
        """
        Calculate great circle distance using haversine formula
        """
        # Convert to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return self.EARTH_RADIUS * c

    def find_nearest_neighbors(self, catalog: Any, params: Dict[str, float], 
                             verbose: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate nearest neighbor distances in space-time-magnitude domain
        """
        rmax = kwargs.get('rmax', 500)  # km
        tmax = kwargs.get('tmax', 20)   # years
        M0 = kwargs.get('M0', 0)        # reference magnitude
        
        # Initialize arrays
        event_count = catalog.size()
        nn_distances = np.zeros(event_count)
        parent_ids = np.zeros(event_count)
        child_ids = np.zeros(event_count)

        # Add small uncertainty if requested
        if kwargs.get('correct_colocated', False):
            catalog.data['Lon'] += np.random.randn(event_count) * 1e-10

        for i in range(event_count):
            if verbose:
                print(f"Processing event {i+1} of {event_count}", end="\r")

            # Calculate time differences
            time_diff = catalog.data['Time'][i] - catalog.data['Time']
            potential_parents = time_diff > 0

            if np.sum(potential_parents) > 0:
                curr_parents = np.where(potential_parents)[0]
                
                # Calculate spatial distances
                if all(k in catalog.data for k in ['X', 'Y']):
                    distances = np.sqrt(
                        (catalog.data['X'][i] - catalog.data['X'][curr_parents])**2 +
                        (catalog.data['Y'][i] - catalog.data['Y'][curr_parents])**2
                    )
                else:
                    distances = self.calculate_haversine_distance(
                        catalog.data['Lon'][i], catalog.data['Lat'][i],
                        catalog.data['Lon'][curr_parents], catalog.data['Lat'][curr_parents]
                    )

                # Filter by maximum distance
                close_events = distances < rmax
                if np.sum(close_events) > 0:
                    valid_parents = curr_parents[close_events]
                    valid_distances = distances[close_events]
                    
                    # Calculate combined metric
                    mag_diff = catalog.data['Mag'][valid_parents] - M0
                    eta = (time_diff[valid_parents] * 
                          (valid_distances**params['D']) * 
                          (10**(-params['b'] * mag_diff)))
                    
                    # Find minimum eta
                    min_idx = np.argmin(eta)
                    nn_distances[i] = eta[min_idx]
                    parent_ids[i] = catalog.data['N'][valid_parents[min_idx]]
                    child_ids[i] = catalog.data['N'][i]

        valid_links = nn_distances > 0
        return {
            'distances': nn_distances[valid_links],
            'parent_ids': parent_ids[valid_links],
            'child_ids': child_ids[valid_links],
            'times': catalog.data['Time'][valid_links]
        }

    def create_clusters(self, nn_data: Dict[str, np.ndarray], 
                       similarity_threshold: float, 
                       verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Group events into clusters based on nearest neighbor distances
        """
        # Remove identical parent-child pairs
        unique_pairs = abs(nn_data['child_ids'] - nn_data['parent_ids']) > 0
        filtered_data = {k: v[unique_pairs] for k, v in nn_data.items()}
        
        # Sort by time
        time_order = np.argsort(filtered_data['times'])
        sorted_data = {k: v[time_order] for k, v in filtered_data.items()}
        
        # Initialize clusters
        clusters = {'0': np.array([])}  # Singles cluster
        cluster_count = 1
        
        # Process each event pair
        for i in range(len(sorted_data['distances'])):
            if sorted_data['distances'][i] >= similarity_threshold:
                continue
                
            child_id = sorted_data['child_ids'][i]
            parent_id = sorted_data['parent_ids'][i]
            
            # Find existing cluster
            found_cluster = False
            for cluster_id in list(clusters.keys())[1:]:  # Skip singles
                if parent_id in clusters[cluster_id] or child_id in clusters[cluster_id]:
                    clusters[cluster_id] = np.unique(
                        np.append(clusters[cluster_id], [parent_id, child_id])
                    )
                    found_cluster = True
                    break
            
            # Create new cluster if needed
            if not found_cluster:
                clusters[str(cluster_count)] = np.array([parent_id, child_id])
                cluster_count += 1

        # Merge overlapping clusters
        i = 0
        while i < len(clusters):
            merged = False
            cluster_i = clusters[str(i)]
            
            for j in range(i + 1, len(clusters)):
                cluster_j = clusters[str(j)]
                if np.intersect1d(cluster_i, cluster_j).size > 0:
                    # Merge clusters
                    clusters[str(i)] = np.unique(np.concatenate([cluster_i, cluster_j]))
                    clusters.pop(str(j))
                    merged = True
                    break
            
            if not merged:
                i += 1

        if verbose:
            total_events = sum(len(cluster) for cluster in clusters.values())
            print(f"Created {len(clusters)-1} clusters with {total_events} total events")
        
        return clusters

    def calculate_rescaled_parameters(self, child_catalog, parent_catalog, params, **kwargs):
        """
        Calculate rescaled time and distance parameters.
        Follows original algorithm from GitHub's clustering.py
        """
        # Get reference magnitude
        M0 = kwargs.get('M0', 0)
        
        # Add uncertainty if requested
        if kwargs.get('correct_colocated', False):
            uncertainty = np.random.randn(child_catalog.size()) * 1e-10
            child_catalog.data['Lon'] += uncertainty

        # Calculate magnitude correction factor
        mag_correction = 10**(-0.5 * params['b'] * (parent_catalog.data['Mag'] - M0))

        # Calculate distances
        if all(k in child_catalog.data for k in ['X', 'Y']):
            distances = np.sqrt(
                (child_catalog.data['X'] - parent_catalog.data['X'])**2 +
                (child_catalog.data['Y'] - parent_catalog.data['Y'])**2
            )
            rescaled_distances = distances**params['D'] * mag_correction
        else:
            distances = self.calculate_haversine_distance(
                child_catalog.data['Lon'], child_catalog.data['Lat'],
                parent_catalog.data['Lon'], parent_catalog.data['Lat']
            )
            rescaled_distances = distances**params['D'] * mag_correction

        # Calculate time differences and rescaled times
        time_diffs = child_catalog.data['Time'] - parent_catalog.data['Time']
        rescaled_times = time_diffs * mag_correction

        # Check for invalid time order
        invalid_times = rescaled_times < 0
        if np.any(invalid_times):
            error_msg = f"{np.sum(invalid_times)} parents occurred after offspring, "
            error_msg += "check order of origin time in child_catalog, parent_catalog"
            raise ValueError(error_msg)

        return rescaled_distances, rescaled_times
    
    def generate_random_catalog(self, catalog: Any, time_range: Tuple[float, float]) -> None:
        """
        Generate randomized catalog maintaining same statistics
        """
        catalog.data['Time'] = np.random.uniform(
            time_range[0], time_range[1], 
            size=catalog.size()
        )
        catalog.sort_catalog('Time')
