from typing import Optional, Iterable, Tuple
import numpy as np
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
import xarray as xr
from rs_tools._src.geoprocessing.geometry import bbox_string_to_bbox
from odc.geo.geom import Geometry, BoundingBox


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


def rioxarray_resample(
    ds: xr.Dataset,
    resample_algorithm = Resampling.bilinear,
    resolution: float = 1_000,
    
):
    return ds.rio.reproject(ds.rio.crs, resolution=resolution, resampling=resample_algorithm, )


def rioxarray_clip_from_bbox(
    ds: xr.Dataset,
    bbox: BoundingBox,
):
    return ds.rio.clip_box(*bbox.bbox, crs=bbox.crs)


def rioxarray_clip_from_polygon(
    ds: xr.Dataset,
    geometry: Geometry,
    all_touched: bool=True,
    drop: bool=True,
):
    return ds.rio.clip(geometries=[geometry.polygon], crs=geometry.crs, all_touched=all_touched, drop=drop)
