from typing import Optional, List, Iterable, Tuple, Callable, Any, cast
from rtree.index import Index, Property
from pathlib import Path
import pandas as pd
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.polygon import Polygon
from rasterio.crs import CRS
import rioxarray
import xarray as xr
import numpy as np
import re
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from odc.geo.geom import BoundingBox, Geometry
# from torchgeo.datasets.utils import BoundingBox, disambiguate_timestamp
from rs_tools._src.modules.goes16.io import GOES16FileOrg, GOESFileName
from rs_tools._src.modules.goes16.meta import GOES16_BANDS_TO_WAVELENGTHS
from rs_tools._src.preprocessing.nans import check_nan_count
from rs_tools._src.geoprocessing.reproject import calculate_latlon
from rs_tools._src.utils.io import get_list_filenames
from loguru import logger


def load_raw_single_band(
    full_path,
    bbox: BoundingBox | None = None,
    crs: CRS | None = None,
    res: Tuple[float, float] | float | None = None,
    resampling: Resampling = Resampling.bilinear,
    fill_value: float = 0.0,
    ) -> xr.Dataset:
    """
    Load a single band from a dataset.

    Args:
        full_path (str): The full path to the dataset.
        bbox (BoundingBox | None, optional): The bounding box coordinates (left, bottom, right, top) to clip the data. Defaults to None.
        crs (CRS | None, optional): The coordinate reference system to use for clipping and reprojection. Defaults to None.
        res (float | None, optional): The desired resolution for reprojection. Defaults to None.
        resampling (Resampling, optional): The resampling method to use for reprojection. Defaults to Resampling.bilinear.
        fill_value (float, optional): The value to use for filling missing or masked data. Defaults to 0.0.

    Returns:
        xr.Dataset: The loaded band as a Dataset object.

    Raises:
        ValueError: If the filename does not conform to the GOES16 Standard.

    """
    
    try:
        goes_filename = GOESFileName.init_from_filename(full_path)
        band = goes_filename.channel
        time = goes_filename.time_start
    except ValueError:
        msg = f"Unrecognized filename: \n{Path(full_path).name}"
        msg += f"\nDoes not conform to GOES16 Standard..."
        raise ValueError(msg)
    # open the dataset
    xdset = rioxarray.open_rasterio(full_path)
    
    if crs is None:
        crs = xdset.rio.crs
    
    # clip data
    if bbox is not None:
        xdset = xdset.rio.clip_box(*bbox.bbox, crs=bbox.crs)

    # reproject
    if res is not None: 
        xdset = xdset.rio.reproject(dst_crs=crs, resolution=res, resampling=resampling)
        
    # extract meta-data from filename
    
    
    # add the band dimensions
    xdset = xdset.assign_coords({"band": [band]})
    xdset["band"].attrs = {
        "long_name": "ABI band number",
        "standard_name": "sensor_band_identifier"
    }
    
    # add the time dimension
    xdset = xdset.expand_dims({"time": [time]})
    
    band_str = str(int(band[1:]))
    xdset = xdset.assign_coords({"band_wavelength": (("band"), [GOES16_BANDS_TO_WAVELENGTHS[band_str]])})
    xdset["band_wavelength"].attrs = {
        "long_name": "ABI band central wavelength",
        "standard_name": "sensor_band_central_radiation_wavelength",
        "units": "um"
    }
    
    # convert fill value
    xdset["Rad"] = xdset["Rad"].where(xdset["Rad"] != xdset["Rad"].attrs["_FillValue"], fill_value)
    
    # fix attributes
    keys = ["long_name", "standard_name", "valid_range", "units", "scale_factor", "add_offset", "fill_value"]
    keep_attrs = {key: xdset.Rad.attrs[key] for key in keys if key in xdset["Rad"].attrs.keys()}
    xdset["Rad"].attrs = {}
    xdset["Rad"].attrs = keep_attrs
    
    # move the data-quality flags to coordinates
    xdset = xdset.drop_vars("DQF")
    
    # just in case...
    xdset.rio.write_crs(crs, inplace=True)

    return xdset.Rad


def load_raw_stacked_band(
    full_paths: List[str],
    bbox: BoundingBox | None = None,
    crs: CRS | None = None,
    res: Tuple[float, float] | float | None = None,
    resampling: Resampling = Resampling.bilinear,
    fill_value: float = 0.0,
    transforms: Callable[[Any], Any] | None = None,
    post_transforms: Callable[[Any], Any] | None = None,
    ) -> xr.Dataset:
    
    assert len(full_paths) <= 16
    bands = []
    band_wavelengths = []
    goes_files = list(map(lambda x: GOESFileName.init_from_filename(x), full_paths))
    
    for i, igoes_files in enumerate(goes_files):
        # open dataset
        xdset = load_raw_single_band(
            igoes_files.full_path,
            bbox=bbox,
            crs=crs,
            res=res,
            resampling=resampling,
            fill_value=fill_value,
            )
        bands.append(xdset.band.values.squeeze())
        band_wavelengths.append(xdset.band_wavelength.values.squeeze())
        band_wl_attrs = xdset.band_wavelength.attrs
        band_attrs = xdset.band.attrs
        # apply transforms
        if transforms:
            xdset = transforms(xdset)
        
        # concatenation
        if i == 0:
            stacked_ds = xdset
            
        else:
            xdset = xdset.interp(x=stacked_ds.x, y=stacked_ds.y)
            stacked_ds = xr.concat([stacked_ds, xdset], dim="band")
        
        # del xdset
        
    # reassign bands
    dims = stacked_ds.dims
    stacked_ds = stacked_ds.assign_coords({"band": bands})
    stacked_ds["band"].attrs = band_attrs
    stacked_ds = stacked_ds.assign_coords({"band_wavelength": (("band"), band_wavelengths)})
    stacked_ds["band_wavelength"].attrs = band_wl_attrs
    if "time" not in dims:
        stacked_ds = stacked_ds.expand_dims({"time": [igoes_files.time_start]})
        
    # apply transforms
    if post_transforms:
        stacked_ds = post_transforms(stacked_ds)
    
    return stacked_ds


def save_g16_band_to_tiff(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool=False,
        save_latlon: bool=True,
        fill_value: float=0.0
):  
    """
    Save a GOES-16 band to a TIFF file.

    Parameters:
        ds (xr.DataArray): The input data array containing the band data.
        save_dir (str): The directory where the TIFF file will be saved.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
        save_latlon (bool, optional): Whether to save latitude and longitude coordinates. Defaults to True.
        fill_value (float, optional): The fill value to use for NaN values. Defaults to 0.0.

    Returns:
        None
    """
    ifile_path = Path(save_dir)
    
    # assert directory exists
    ifile_path.mkdir(parents=True, exist_ok=True)
    
    # grab time components
    time_stamp = pd.to_datetime(ds.time.values.squeeze())
    save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
    band = ds.band.values
    if len(band) > 1:
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_rads.tiff")
    else:
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_{band.squeeze()}_rads.tiff")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()
    # select variables
    
    crs = ds.rio.crs
    ds = ds.where(ds != np.nan, fill_value)
    # remove all extra dimensions
    ds = ds.squeeze()
    
    logger.info(f"Dims: {ds.dims}")
    # assert len(ds.dims) == 3
    
    
    
    # add coordinates
    if save_latlon:
        ds = ds.to_dataset(dim="band")
        LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
        ds["lat"] = (("x", "y", "band"), LATS[..., None])
        ds["lon"] = (("x", "y", "band"), LONS[..., None])
        ds = ds.to_array(dim="band")
        
        
    # change long name for rasterio to understand the band descriptions...
    try:
        ds.attrs["long_name"] = list(map(lambda x: str(x), ds.band.values))
    except TypeError:
        ds.attrs["long_name"] = str(ds.band.values)
    logger.info(f"long_name: {ds.attrs['long_name']}")

    # save with rasterio
    ds.rio.to_raster(ifile_path)

    return None
    

def save_g16_to_tiff(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool=False,
        save_latlon: bool=True,
        fill_value: float=0.0
):  
        ifile_path = Path(save_dir)
        
        # assert directory exists
        ifile_path.mkdir(parents=True, exist_ok=True)
        
        # grab time components
        time_stamp = pd.to_datetime(ds.time.values.squeeze())
        save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
        
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_rads.tiff")

        if overwrite and ifile_path.is_file():
            # remove file if it already exists
            ifile_path.unlink()
        # select variables
        
        crs = ds.rio.crs
        ds = ds
        ds = ds.where(ds != np.nan, fill_value)
        # create a dataset
        ds = ds.to_dataset(dim="band")
        
        ds.rio.write_crs(crs, inplace=True)

        # remove attributes (otherwise it doesnt save properly...)
        ds.attrs = {}

        # add coordinates
        if save_latlon:
            LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
            ds["lat"] = (("x", "y"), LATS)
            ds["lon"] = (("x", "y"), LONS)
        
        # save with rasterio
        ds.squeeze().rio.to_raster(ifile_path)

        return None

    
def save_g16_band_to_npy(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool=False,
        save_latlon: bool=True,
        fill_value: float=0.0
        ):
      
        ifile_path = Path(save_dir)
        
        # assert directory exists
        ifile_path.mkdir(parents=True, exist_ok=True)
        
        # grab time components
        time_stamp = pd.to_datetime(ds.time.values.squeeze())
        save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
        band = ds.band.values.squeeze()
        
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_{band}_rads.tiff")

        if overwrite and ifile_path.is_file():
            # remove file if it already exists
            ifile_path.unlink()
        # select variables
        
        crs = ds.rio.crs
        ds = ds.Rad
        ds = ds.where(ds != np.nan, fill_value)
        # create a dataset
        ds = ds.to_dataset(dim="band")
        
        ds.rio.write_crs(crs, inplace=True)

        # remove attributes (otherwise it doesnt save properly...)
        ds.attrs = {}

        # add coordinates
        if save_latlon:
            LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
            ds["lat"] = (("x", "y"), LATS)
            ds["lon"] = (("x", "y"), LONS)
        
        # save with rasterio
        ds.squeeze().rio.to_raster(ifile_path)

        return None
    
