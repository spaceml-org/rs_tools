from typing import Union, List, Dict, Tuple
import numpy as np
import xarray as xr

def check_sat_FOV(region: Tuple[int, int, int, int], FOV: Tuple[int, int, int, int]) -> bool:
    """
    Check if the region is within the Field of View (FOV) of the satellite.
    
    Args:
        region (Tuple[int, int, int, int]): The region (lon_min, lat_min, lon_max, lat_max) to check if it is within the FOV
        FOV (Tuple[int, int]): The Field of View (FOV) (lon_min, lat_min, lon_max, lat_max) of the satellite.
    
    Returns:
        bool: True if the region is within the FOV, False otherwise.
    """
    # Check if the region is within the Field of View (FOV) of the satellite.
    if abs(region[0]) <= abs(FOV[0]) and abs(region[1]) <= abs(FOV[1]) and abs(region[2]) <= abs(FOV[2]) and abs(region[3]) <= abs(FOV[3]):
        return True
    else:
        return False

def convert_units(ds: xr.Dataset, wavelengths: Dict) -> xr.Dataset:
    """
    Function to convert units from mW/m^2/sr/cm^-1 to W/m^2/sr/um.
    Acts on each band separately.
    
    Parameters:
        ds (xr.Dataset): The input dataset to be converted.
        wavelengths (Dict[float]): Dictionary of wavelengths of data for each band (i).
        
    Returns:
        xr.Dataset: The converted dataset.
    """
    for band in ds.data_vars:
        ds[band] = ds[band] * 0.001  # to convert mW to W
        ds[band] = ds[band] * 10000 / wavelengths[band]**2  # to convert cm^-1 to um
    return ds
