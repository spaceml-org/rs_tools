import xarray as xr
import pandas as pd
from satpy import Scene
from pathlib import Path
import numpy as np
from rs_tools._src.data.modis.bands import get_modis_channel_numbers, CALIBRATION_CHANNELS, MODIS_WAVELENGTHS, MODIS_CHANNEL_WAVELENGTHS_CENTRAL
from rs_tools._src.data.modis.variables import VARIABLE_ATTRS
from loguru import logger
from datetime import datetime
from rs_tools._src.geoprocessing.reproject import calculate_latlon


def load_modis_bands_raw(file: str, calibration: str = "radiance") -> xr.Dataset:
    """
    Load MODIS data from a file and return it as an xarray Dataset.

    Parameters:
        file (str): The path to the MODIS data file.
        calibration (str, optional): The calibration type. Defaults to "radiance".

    Returns:
        xr.Dataset: The MODIS data as an xarray Dataset.
    """

    # load file with satpy
    scn = Scene(
        reader="modis_l1b",
        filenames=[
            str(file),
        ],
    )
    # get channels
    wishlist = CALIBRATION_CHANNELS[calibration]

    # load image
    scn.load(wishlist, generate=False, calibration=calibration)

    # convert to xarray
    ds = scn.to_xarray_dataset(datasets=wishlist)

    return ds


def load_modis_cloud_mask_raw(file: str) -> xr.Dataset:
    """
    Load MODIS cloud mask data from a file and return it as an xarray Dataset.

    Parameters:
        file (str): The path to the MODIS cloud mask file.

    Returns:
        xr.Dataset: An xarray Dataset containing the loaded cloud mask data.

    """
    # Load file using satpy scenes
    scn = Scene(
        reader="modis_l2",
        filenames=[
            str(file),
        ],
    )
    # Load cloud mask data
    datasets = scn.available_dataset_names()

    # Note: Needs to be loaded at 1000 m resolution for all channels to match
    scn.load(datasets, generate=False, resolution=1_000)

    # change to xarray data
    ds = scn.to_xarray_dataset()
    ds = ds.assign_coords({"longitude": ds["cloud_mask"].area.lons})
    ds = ds.assign_coords({"latitude": ds["cloud_mask"].area.lats})

    return ds


def preprocess_modis_raw(ds: xr.Dataset, calibration: str="radiance") -> xr.Dataset:
    """
    Preprocesses MODIS image radiances.

    Args:
        ds (xr.Dataset): The input MODIS image dataset.
        calibration (str, optional): The calibration type. Defaults to "radiance".

    Returns:
        xr.Dataset: The preprocessed MODIS image radiances as an xarray dataset.
    """

    # grab variable names from list of dataset variables
    channels = list(ds.data_vars)

    # Store the attributes in a dict before concatenation
    attrs_dict = {x: ds[x].attrs for x in channels}

    # concatinate in new band dimension, and defining a new variable name
    # NOTE: Concatination overwrites attrs of bands.
    ds = ds.assign({str(calibration): xr.concat(list(map(lambda x: ds[x], channels)), dim="band")})
    
    # drop duplicate variables
    ds = ds.drop_vars(list(map(lambda x: x, channels)))

    # ================
    # COORDINATES
    # ================
    # rename band dimensions
    ds = ds.assign_coords(band=list(map(lambda x: x, channels)))

    # convert measurement time (in seconds) to datetime
    time_stamp = pd.to_datetime(ds.attrs["start_time"])
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")
    # assign bands and time data to each variable
    ds = ds.assign_coords({"time": [time_stamp]})

    # assign band wavelengths
    wavelengths = [MODIS_CHANNEL_WAVELENGTHS_CENTRAL[str(iband)] for iband in ds.band.values]
    ds = ds.assign_coords({"band_wavelength": (("band"), wavelengths)})

    # ================
    # ATTRIBUTES
    # ================

    # NOTE: Keep only certain relevant attributes
    ds.attrs = {}
    ds[calibration].attrs = {}
    ds[calibration].attrs = VARIABLE_ATTRS[calibration]
    ds[calibration].attrs["platform_name"] = attrs_dict[list(attrs_dict.keys())[0]]["platform_name"]
    ds[calibration].attrs["sensor"] = attrs_dict[list(attrs_dict.keys())[0]]["sensor"]

    return ds


def preprocess_cloud_mask_raw(ds: xr.Dataset, calibration: str="cloud_mask") -> xr.Dataset:
    """
    Preprocesses MODIS image radiances.

    Args:
        ds (xr.Dataset): The input MODIS image radiances as an xarray dataset.
        calibration (str, optional): The calibration type. Defaults to "cloud_mask".

    Returns:
        xr.Dataset: The preprocessed MODIS image radiances as an xarray dataset.
    """
    attrs_dict = ds[calibration].attrs

    # convert measurement time (in seconds) to datetime
    time_stamp = pd.to_datetime(ds.attrs["start_time"])
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")
    # assign bands and time data to each variable
    ds = ds.assign_coords({"time": [time_stamp]})


    # ================
    # ATTRIBUTES
    # ================

    # NOTE: Keep only certain relevant attributes
    ds.attrs = {}
    
    ds[calibration].attrs = {}
    ds[calibration].attrs = VARIABLE_ATTRS[calibration]
    ds[calibration].attrs["platform_name"] = attrs_dict["platform_name"]
    ds[calibration].attrs["sensor"] = attrs_dict["sensor"]
    
    ds = ds.drop_vars("quality_assurance")

    return ds


def save_modis_band_to_tiff(
        ds: xr.DataArray,
        calibration: str,
        save_dir: str,
        overwrite: bool=False,
        save_latlon: bool=False,
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
    
    
    # extract satellite from meta-data
    satellite = ds[calibration].attrs["platform_name"].split("-")[1].lower()
    
    # grab time components
    time_stamp = pd.to_datetime(ds.time.values.squeeze())
    save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
    ifile_path = ifile_path.joinpath(f"{save_file_name}_modis_{satellite}_{calibration}.tiff")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()
    # select variables
    
    crs = ds.rio.crs
    ds = ds[calibration]
    ds = ds.where(ds != np.nan, fill_value)
    # remove all extra dimensions
    ds = ds.squeeze()
    
    logger.info(f"Dims: {ds.dims}")
    # assert len(ds.dims) == 3
    
    
    
    # add coordinates
    if save_latlon:
        ds = ds.to_dataset(dim="band")
        LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
        ds["lat"] = (("longitude", "latitude", "band"), LATS[..., None])
        ds["lon"] = (("longitude", "latitude", "band"), LONS[..., None])
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


def save_modis_cloud_mask_to_tiff(
        ds: xr.DataArray,
        save_dir: str,
        calibration: str = "cloud_mask",
        overwrite: bool = False,
        save_latlon: bool = False,
        fill_value: float = 0.0
):  
    """
    Save a MODIS cloud mask to a TIFF file.

    Parameters:
        ds (xr.DataArray): The input data array containing the MODIS data.
        save_dir (str): The directory where the TIFF file will be saved.
        calibration (str, optional): The calibration to use for saving the cloud mask. Defaults to "cloud_mask".
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
        save_latlon (bool, optional): Whether to save latitude and longitude coordinates. Defaults to False.
        fill_value (float, optional): The fill value to use for NaN values. Defaults to 0.0.

    Returns:
        None
    """
    ifile_path = Path(save_dir)
    
    # assert directory exists
    ifile_path.mkdir(parents=True, exist_ok=True)
    
    
    # extract satellite from meta-data
    satellite = ds[calibration].attrs["platform_name"].split("-")[1].lower()
    
    # grab time components
    time_stamp = pd.to_datetime(ds.time.values.squeeze())
    save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
    ifile_path = ifile_path.joinpath(f"{save_file_name}_modis_{satellite}_{calibration}.tiff")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()
    # select variables
    
    crs = ds.rio.crs
    ds = ds[calibration]
    ds = ds.where(ds != np.nan, fill_value)
    # remove all extra dimensions
    ds = ds.squeeze()
    
    logger.info(f"Dims: {ds.dims}")
    # assert len(ds.dims) == 3
    
    
    
    # add coordinates
    if save_latlon:
        ds = ds.to_dataset(dim="band")
        LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
        ds["lat"] = (("longitude", "latitude", "band"), LATS[..., None])
        ds["lon"] = (("longitude", "latitude", "band"), LONS[..., None])
        ds = ds.to_array(dim="band")
        
        
    # change long name for rasterio to understand the band descriptions...
    ds.attrs["long_name"] = "cloud_mask"

    # save with rasterio
    ds.rio.to_raster(ifile_path)

    return None

