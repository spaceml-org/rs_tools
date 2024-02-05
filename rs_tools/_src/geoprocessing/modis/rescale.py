from dataclasses import dataclass
import xarray as xr
import numpy as np

def convert_integers2radiances(
        da: xr.DataArray, 
    ) -> xr.DataArray:
    """
    Function to convert scaled integers to radiances.
    Scaled integers are unitless, radiances are given in W^2/m^2/um/sr
    """
    
    radiance_scale = da.radiance_scales
    radiance_offsets = da.radiance_offsets 

    radiance_scale = np.expand_dims(radiance_scale, axis=(1,2))
    radiance_offsets = np.expand_dims(radiance_offsets, axis=(1,2))

    assert radiance_offsets.shape == radiance_scale.shape

    corrected_data = (da - radiance_offsets)*radiance_scale

    #TODO - change attributes to have units
    #TODO - change attributes to change name of variable
    #TODO - change attributes to change rescaling

    return corrected_data


def convert_integers2reflectances(
        da: xr.DataArray, 
    ) -> xr.DataArray:
    """
    Function to convert scaled integers to reflectances.
    Scaled integers and reflectances are both unitless.
    """
    
    reflectance_scale = da.reflectance_scales
    reflectance_offsets = da.reflectance_offsets 

    reflectance_scale = np.expand_dims(reflectance_scale, axis=(1,2))
    reflectance_offsets = np.expand_dims(reflectance_offsets, axis=(1,2))

    assert reflectance_offsets.shape == reflectance_scale.shape

    corrected_data = (da - reflectance_offsets)*reflectance_scale

    return corrected_data