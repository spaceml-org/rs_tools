from typing import Union, List, Dict
import numpy as np
import xarray as xr

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