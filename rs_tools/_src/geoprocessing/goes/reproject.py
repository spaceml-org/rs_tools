import xarray as xr
import rioxarray
from pyproj import CRS


def reproject_goes16(ds: xr.Dataset, crs_projection: str="EPSG:4326") -> xr.Dataset:
    """
    Reprojects a GOES-16 dataset to a desired coordinate system.

    Parameters:
        ds (xr.Dataset): The input dataset to be reprojected.
        crs_projection (str): The desired coordinate system for reprojection. Default is "EPSG:4326".

    Returns:
        xr.Dataset: The reprojected dataset.
    """

    # transpose data
    try:
        ds = ds.transpose("band", "y", "x")
    except ValueError:
        pass
        # ds = ds.transpose("y", "x")
        

    # get perspective height
    sat_height = ds.goes_imager_projection.attrs["perspective_point_height"]

    # reassign coordinates to correct height
    ds = ds.assign_coords({"x": ds.x.values * sat_height})
    ds = ds.assign_coords({"y": ds.y.values * sat_height})

    # load CRS
    cc = CRS.from_cf(ds.goes_imager_projection.attrs)

    # assign CRS to dataarray
    ds.rio.write_crs(cc.to_string(), inplace=True)

    # reproject to desired coordinate system
    ds = ds.rio.reproject(crs_projection)
    
    return ds