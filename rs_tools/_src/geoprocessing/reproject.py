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


def calculate_latlon(x, y, crs):
    """
    Calculate the latitude and longitude coordinates for given x and y coordinates in a specified coordinate reference system (CRS).

    Parameters:
    x (array-like): The x-coordinates.
    y (array-like): The y-coordinates.
    crs (str): The coordinate reference system (CRS) of the input coordinates.

    Returns:
    tuple: A tuple containing the longitude and latitude arrays.

    """
    XX, YY = np.meshgrid(x, y)
    lons, lats = convert_x_y_to_lat_lon(crs, XX, YY)
    lons = np.float32(lons)
    lats = np.float32(lats)
    # Check if lons and lons_trans are close in value
    # Set inf to NaN values
    lons[lons == np.inf] = np.nan
    lats[lats == np.inf] = np.nan
    
    return lons, lats


def rioxarray_resample(
    ds: xr.Dataset,
    resolution: float | Tuple[float, float],
    crs: CRS | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> xr.Dataset:
    """
    Resamples the input xarray dataset to a specified resolution using rasterio.

    Parameters:
    - ds (xr.Dataset): The input xarray dataset to be resampled.
    - resolution (float | Tuple[float, float]): The target resolution for resampling. Can be a single value for equal x and y resolutions, or a tuple of x and y resolutions.
    - resampling (Resampling, optional): The resampling method to be used. Defaults to Resampling.bilinear.

    Returns:
    - xr.Dataset: The resampled xarray dataset.

    """
    variables = list(ds.coords.keys())
    non_spatial_dims = set(ds.dims) - set([ds.rio.x_dim, ds.rio.y_dim])
    
    ds = ds.squeeze().rio.reproject(dst_crs=crs, resolution=resolution, resampling=resampling)
    
    for idim in non_spatial_dims:
        ds = ds.expand_dims({idim: [ds[idim].values.squeeze()]})
        
    if "band_wavelength" in variables:
        ds["band_wavelength"] = ds["band_wavelength"].expand_dims("band")
    
    return ds


def rioxarray_clip_from_bbox(ds: xr.Dataset, bbox: BoundingBox) -> xr.Dataset:
    """
    Clips the given xarray dataset to the specified bounding box.

    Parameters:
        ds (xr.Dataset): The xarray dataset to be clipped.
        bbox (BoundingBox): The bounding box to clip the dataset to.

    Returns:
        xr.Dataset: The clipped xarray dataset.
    """
    return ds.rio.clip_box(*bbox.bbox, crs=bbox.crs)


def rioxarray_clip_from_polygon(
    ds: xr.Dataset,
    geometry: Geometry,
    all_touched: bool = True,
    drop: bool = True,
):
    """
    Clips the given xarray Dataset `ds` using the provided polygon `geometry`.

    Parameters:
        ds (xr.Dataset): The xarray Dataset to be clipped.
        geometry (Geometry): The polygon geometry used for clipping.
        all_touched (bool, optional): Whether to include all pixels touched by the polygon. Defaults to True.
        drop (bool, optional): Whether to drop variables that are completely outside the polygon. Defaults to True.

    Returns:
        xr.Dataset: The clipped xarray Dataset.
    """
    return ds.rio.clip(geometries=[geometry.polygon], crs=geometry.crs, all_touched=all_touched, drop=drop)
