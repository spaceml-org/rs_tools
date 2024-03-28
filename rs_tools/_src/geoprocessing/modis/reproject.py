import xarray as xr
import rioxarray
from pyproj import CRS


def add_modis_crs(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds the Coordinate Reference System (CRS) to the given MODIS dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset to which the CRS will be added.

    Returns:
    - xarray.Dataset: The dataset with the CRS added.
    """
    # define CRS of MODIS dataset
    crs = 'WGS84'

    # load source CRS from the WKT string
    cc = CRS(crs)

    # assign CRS to dataarray
    ds.rio.write_crs(cc, inplace=True)

    return ds