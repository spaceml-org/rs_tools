# populates the search tree
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from typing import Tuple


rioxarray_samplers = {
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "nearest": Resampling.nearest,
}

# NOTE: This function has been copied outside of the satellite-specific folders
def resample_rioxarray(ds: xr.Dataset, resolution: Tuple[int, int]=(1_000, 1_000), method: str="bilinear") -> xr.Dataset:
    """
    Resamples a raster dataset using rasterio-xarray.

    Parameters:
        ds (xr.Dataset): The input dataset to be resampled.
        resolution (int): The desired resolution of the resampled dataset. Default is 1_000.
        method (str): The resampling method to be used. Default is "bilinear".

    Returns:
        xr.Dataset: The resampled dataset.
    """

    ds = ds.rio.reproject(
        ds.rio.crs,
        resolution=resolution,
        resample=rioxarray_samplers[method], 
    )
    return ds
