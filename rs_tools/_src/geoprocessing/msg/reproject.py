import xarray as xr
import rioxarray
from pyproj import CRS


def add_msg_crs(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds the Coordinate Reference System (CRS) to the given MSG dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset to which the CRS will be added.

    Returns:
    - xarray.Dataset: The dataset with the CRS added.
    """
    # the CRS is stored in data variable attributes of "msg_seviri_fes_3km"
    var = "msg_seviri_fes_3km"
    crs_wkt = ds[var].crs_wkt

    # load source CRS from the WKT string
    cc = CRS(crs_wkt)

    # assign CRS to dataarray
    ds.rio.write_crs(cc, inplace=True)

    return ds