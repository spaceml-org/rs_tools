from typing import List
import xarray as xr
from pathlib import Path
from functools import partial


def spatial_mean(ds: xr.Dataset, spatial_variables: List[str]) -> xr.Dataset:
    return ds.mean(spatial_variables)

# TODO: Check this function
def normalize(
        files: List[str],
        temporal_variables: List[str], 
        spatial_variables: List[str]=["x","y"], 
) -> xr.Dataset:
    
    preprocess = partial(spatial_mean, spatial_variables=spatial_variables)

    # calculate mean
    ds_mean = xr.open_mfdataset(files, preprocess=preprocess, combine="by_coords",  engine="netcdf4")

    ds_mean = ds_mean.mean(temporal_variables)

    def preprocess(ds):
        # calculate the mean
        ds = ((ds - ds_mean)**2).std([spatial_variables])
        return ds
    
    ds_std = xr.open_mfdataset(files, preprocess=preprocess, combine="by_coords",  engine="netcdf4")

    ds_std = ds_std.mean(temporal_variables)

    ds = xr.combine_by_coords([ds_mean, ds_std])
    return ds







