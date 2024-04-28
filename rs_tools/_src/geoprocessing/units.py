from typing import Union, List, Dict, Tuple
import numpy as np
import xarray as xr
import pandas as pd

def convert_units_ds(ds: xr.Dataset, wavelengths: Dict) -> xr.Dataset:
    """
    Function to convert units from mW/m^2/sr/cm^-1 to W/m^2/sr/um in xarray dataset.
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


def convert_units(data: np.array, wavelengths: np.array) -> np.array:
    """
    Function to convert units from mW/m^2/sr/cm^-1 to W/m^2/sr/um in numpy array.
    Acts on each band separately.
    
    Parameters:
        data (np.array): The input data to be converted.
        wavelengths (np.array): The wavelengths of the input data.
        
    Returns:
        np.array: The converted data.
    """
    assert len(data) == len(wavelengths)
    corrected_data = []
    for i, wvl in enumerate(wavelengths):
        corr_data = data[i] * 0.001 # to convert mW to W
        corr_data = corr_data * 10000 / wvl**2 # to convert cm^-1 to um
        corrected_data.append(corr_data)
    return np.stack(corrected_data, axis=0)


