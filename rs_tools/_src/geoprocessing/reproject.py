from typing import Tuple
import numpy as np
from pyproj import CRS, Transformer
import xarray as xr

# TODO: To be moved to Earth System Datacube Tools
def convert_lat_lon_to_x_y(crs: str, lon: list[float], lat: list[float]) -> Tuple[float, float]:
    transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y


def convert_x_y_to_lat_lon(crs: str, x: list[float], y: list[float]) -> Tuple[float, float]:
    transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def calc_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the latitude and longitude coordinates for the given dataset

    Args:
        ds: Xarray Dataset to calculate the lat/lon coordinates for, with x and y coordinates

    Returns:
        Xarray Dataset with the latitude and longitude coordinates added
    """
    XX, YY = np.meshgrid(ds.x.data, ds.y.data)
    lons, lats = convert_x_y_to_lat_lon(ds.rio.crs, XX, YY)
    lons = np.float32(lons)
    lats = np.float32(lats)
    # Check if lons and lons_trans are close in value
    # Set inf to NaN values
    lons[lons == np.inf] = np.nan
    lats[lats == np.inf] = np.nan

    ds = ds.assign_coords({"latitude": (["y", "x"], lats), "longitude": (["y", "x"], lons)})
    ds.latitude.attrs["units"] = "degrees_north"
    ds.longitude.attrs["units"] = "degrees_east"
    return ds
