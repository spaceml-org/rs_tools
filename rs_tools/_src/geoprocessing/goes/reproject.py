import xarray as xr
import rioxarray
from pyproj import CRS


def add_goes16_crs(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds the Coordinate Reference System (CRS) to the given GOES16 dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset to which the CRS will be added.

    Returns:
    - xarray.Dataset: The dataset with the CRS added.
    """

    # load CRS
    cc = CRS.from_cf(ds.goes_imager_projection.attrs)
    
    # assign CRS to dataarray
    ds.rio.write_crs(cc.to_string(), inplace=True)

    return ds