import xarray as xr
import rioxarray
from pyproj import CRS

GOES16_PYPROJ = "+proj=geos +lon_0=-75 +h=35786023 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True"


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

def add_goes16_crs_satpy(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds the Coordinate Reference System (CRS) to the given GOES dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset to which the CRS will be added.

    Returns:
    - xarray.Dataset: The dataset with the CRS added.
    """
    # define CRS of GOES dataset
    crs = 'GRS80'

    # load source CRS from the WKT string
    cc = CRS(crs)

    # assign CRS to dataarray
    ds.rio.write_crs(cc, inplace=True)

    return ds