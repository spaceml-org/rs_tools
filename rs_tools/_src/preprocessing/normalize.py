from typing import List
import xarray as xr
from pathlib import Path
from functools import partial
import numpy as np


def spatial_mean(ds: xr.Dataset, spatial_variables: List[str]) -> xr.Dataset:
    return ds.mean(spatial_variables)

# TODO: Check this function
def normalize(
        files: List[str],
        temporal_variables: List[str]=["time"], 
        spatial_variables: List[str]=["x","y"], 
) -> xr.Dataset:
    
    preprocess = partial(spatial_mean, spatial_variables=spatial_variables)

    # calculate mean
    ds_mean = xr.open_mfdataset(files, preprocess=preprocess, combine="by_coords",  engine="netcdf4")

    ds_mean = ds_mean.mean(temporal_variables)

    def preprocess(ds: xr.Dataset):
        # calculate the std
        # ds = ((ds - ds_mean)**2).std(spatial_variables)
        N = ds.x.size * ds.y.size
        ds = np.sqrt(((ds - ds_mean) ** 2).sum(['x','y']) / N)
        return ds
    
    ds_std = xr.open_mfdataset(files, preprocess=preprocess, combine="by_coords",  engine="netcdf4")

    ds_std = ds_std.mean(temporal_variables)

    ds_mean = ds_mean.rename({'Rad':'mean'})
    ds_std = ds_std.rename({'Rad':'std'})

    # Drop any variables that are not used (e.g. DQF for GOES)
    ds_mean = ds_mean.drop_vars([v for v in ds_mean.var() if v not in ['std', 'mean']])
    ds_std = ds_std.drop_vars([v for v in ds_std.var() if v not in ['std', 'mean']])

    ds = xr.combine_by_coords([ds_mean, ds_std])
    return ds







