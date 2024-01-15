import xarray as xr
import rioxarray
from pyproj import CRS


def reproject_goes16(ds: xr.Dataset, crs_projection: str="EPSG:4326") -> xr.Dataset:

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
    ds =  ds.rio.write_crs(cc.to_string(), inplace=False)

    # reproject to desired coordinate system
    ds = ds.rio.reproject(crs_projection)
    
    return ds