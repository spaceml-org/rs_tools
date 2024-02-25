import numpy as np
from typing import Tuple


def create_latlon_grid(region: Tuple[float, float, float, float],
                       resolution: float):
    """
    Create a latitude-longitude grid within the specified region.

    Args:
        region (Tuple[float, float, float, float]): The region of interest defined by
            (lon_min, lat_min, lon_max, lat_max), e.g. (-180, -90, 180, 90) for global grid.
        resolution (float): The resolution of the grid in degrees.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays representing
            the latitudes and longitudes of the grid, respectively.
    """
    
    lat_bnds = region[1], region[3]
    lon_bnds = region[0], region[2]
    latitudes = np.arange(lat_bnds[0], lat_bnds[1]+resolution, resolution)
    longitudes = np.arange(lon_bnds[0], lon_bnds[1]+resolution, resolution)
    return np.meshgrid(latitudes, longitudes)