from typing import (
    Dict,
    Tuple,
)
import warnings
from pathlib import Path
import numpy as np
from odc.geo.crs import CRS
from odc.geo.geom import BoundingBox
import pandas as pd
from rasterio.enums import Resampling
import rioxarray
from satpy import Scene
import xarray as xr
from rs_tools._src.geoprocessing.reproject import calculate_latlon
from rs_tools._src.modules.msg.meta import MSG_BANDS_TO_WAVELENGTHS
from loguru import logger
from datetime import datetime


def load_msg_rads_raw_satpy(file: str) -> xr.Dataset:
    """
    Load MSG radiance data from a file and return it as an xarray Dataset.

    Parameters:
        file (str): The path to the file containing the MSG radiance data.

    Returns:
        xr.Dataset: An xarray Dataset containing the loaded MSG radiance data.
    """

    # Load file using satpy scenes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scn = Scene(reader="seviri_l1b_native", filenames=[file])
        # Load radiance bands
        channels = [x for x in scn.available_dataset_names() if x != "HRV"]
        assert len(channels) == 11, "Number of channels is not 11"

        scn.load(channels, generate=False, calibration="radiance")

        # change to xarray data
        ds = scn.to_xarray()

    return ds


def clean_msg_rad_raw_satpy(
    ds: xr.Dataset,
    bbox: BoundingBox | None = None,
    crs: CRS | None = None,
    res: Tuple[float, float] | float | None = None,
    resampling: Resampling = Resampling.bilinear,
    fill_value: float = 0.0,
) -> xr.Dataset:

    ds = ds.set_coords("msg_seviri_fes_3km")
    variables = set(ds.variables.keys())
    coords = set(ds.coords.keys())
    variables = list(variables - coords)

    attrs_dict = {x: ds[x].attrs for x in variables}

    start_time = pd.to_datetime(filter_nest_dict(attrs_dict, "start_time"))

    src_crs = extract_msg_crs(ds)
    ds.rio.write_crs(src_crs, inplace=True)

    # add coordiante reference system
    if crs is None:
        crs = src_crs

    ds = ds.reset_coords(drop=True)

    ds = ds.assign_coords({"time": start_time})

    ds = ds.to_array(dim="band", name="radiance")

    ds = ds.fillna(fill_value)

    ds.attrs = {}
    ds.attrs = dict(
        calibration=filter_nest_dict(attrs_dict, "calibration"),
        standard_name=filter_nest_dict(attrs_dict, "standard_name"),
        platform_name=filter_nest_dict(attrs_dict, "platform_name"),
        sensor=filter_nest_dict(attrs_dict, "sensor"),
        units=filter_nest_dict(attrs_dict, "units"),
        orbital_parameters=filter_nest_dict(attrs_dict, "orbital_parameters"),
    )

    # assign band wavelength coordinates
    wavelengths = [MSG_BANDS_TO_WAVELENGTHS[iband] for iband in ds.band.values]
    ds = ds.assign_coords({"band_wavelength": (("band"), wavelengths)})

    # rewrite the source CRS
    ds.rio.write_crs(src_crs, inplace=True)

    # clip data
    if bbox is not None:
        ds = ds.rio.clip_box(*bbox.bbox, crs=bbox.crs)

    # reproject
    if res is not None:
        # assign fill value to attribute
        ds.attrs["_FillValue"] = fill_value
        ds = ds.rio.reproject(
            dst_crs=crs, resolution=res, resampling=resampling, fill_value=fill_value
        )
        # fix the fill value
        ds.attrs.pop("_FillValue")

    # add time dimension
    # ds = ds.expand_dims({"time": [start_time]})

    ds = ds.fillna(fill_value)

    ds.attrs["fill_value"] = fill_value

    ds.rio.write_crs(crs, inplace=True)

    return ds


def filter_nest_dict(attrs_dict: Dict, key_value: str) -> str:

    filtered_dict = {
        outer_key: inner_dict[key_value]
        for outer_key, inner_dict in attrs_dict.items()
        if key_value in inner_dict
    }
    key_value = list(set(filtered_dict.values()))[0]

    return key_value


def extract_msg_crs(ds: xr.Dataset) -> xr.Dataset:
    var = "msg_seviri_fes_3km"
    crs_wkt = ds[var].crs_wkt
    # load source CRS from the WKT string
    crs = CRS(crs_wkt)
    return crs


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


def save_msg_scene_to_tiff(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool = False,
        save_latlon: bool = False,
        fill_value: float = 0.0
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
    
    ifile_path = ifile_path.joinpath(f"{save_file_name}_msg_rads.tiff")
    logger.debug(f"Save path: {ifile_path}")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()
    # select variables
    
    crs = ds.rio.crs
    ds = ds.where(ds != np.nan, fill_value)
    # remove all extra dimensions
    
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
    ds.squeeze().rio.to_raster(ifile_path)

    return None
    

def save_msg_scene_to_zarr(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool = False,
        save_latlon: bool = False,
        fill_value: float = 0.0,
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
    
    try:
        ds = ds.expand_dims("time")
    except ValueError:
        pass
        
    
    ifile_path = ifile_path.joinpath(f"{save_file_name}_msg_rads.zarr")
    logger.debug(f"Save path: {ifile_path}")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()
    # select variables
    
    crs = ds.rio.crs
    ds = ds.where(ds != np.nan, fill_value)
    # remove all extra dimensions
    
    logger.info(f"Dims: {ds.dims}")
    # assert len(ds.dims) == 3
    
    
    
    # add coordinates
    if save_latlon:
        LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
        ds = ds.assign_coords({"lon": (("x", "y"), LONS)})
        ds = ds.assign_coords({"lat": (("x", "y"), LATS)})
        
    

    # save with rasterio
    ds.to_zarr(ifile_path, mode="w")
    return None
    

