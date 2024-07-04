import autoroot
import typer
from pathlib import Path
from loguru import logger
import xarray as xr
import pandas as pd
from pyproj import CRS
from rs_tools._src.geoprocessing.goes.validation import correct_goes16_bands, correct_goes16_satheight
from rs_tools._src.geoprocessing.goes.reproject import add_goes16_crs
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.modis import parse_modis_dates_from_file
from rs_tools._src.geoprocessing.match import match_timestamps
from rs_tools._src.data.modis import MODIS_NAME_TO_ID


def clean_coords_goes16(ds):
    # convert measurement angles to horizontal distance in meters
    ds = correct_goes16_satheight(ds) 
    try:
        # correct band coordinates to reorganize xarray dataset
        ds = correct_goes16_bands(ds) 
    except AttributeError:
        pass
    # assign coordinate reference system
    ds = add_goes16_crs(ds)

    return ds


def preprocess_goes16_radiances(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocesses the GOES16 radiance dataset.

    Args:
        ds (xr.Dataset): The input dataset.

    Returns:
        xr.Dataset: The preprocessed dataset.
    """
    variables = ["Rad"]

    # Extract relevant attributes from original dataset
    # # convert measurement time (in seconds) to datetime
    time_stamp = pd.to_datetime(ds.t.values).strftime("%Y-%m-%d %H:%M") 
    band_attributes = ds.band.attrs
    band_wavelength_attributes = ds.band_wavelength.attrs
    band_wavelength_values = ds.band_wavelength.values
    band_values = ds.band.values
    cc = CRS.from_cf(ds.goes_imager_projection.attrs)

    # clean the coordinates
    ds = clean_coords_goes16(ds)
    
    # assign bands data to each variable
    ds = ds[variables]
    ds = ds.expand_dims({"band": band_values})
    # attach time coordinate
    ds = ds.assign_coords({"time": [time_stamp]})
    # drop variables that will no longer be needed
    ds = ds.drop_vars(["t", "y_image", "x_image", "goes_imager_projection"])
    # assign band attributes to dataset
    ds.band.attrs = band_attributes
    # assign band wavelength to each variable
    ds = ds.assign_coords({"band_wavelength": band_wavelength_values})
    ds.band_wavelength.attrs = band_wavelength_attributes

    # write crs
    ds.rio.write_crs(cc.to_string(), inplace=True)

    # Keep only certain relevant attributes
    attrs_rad = ds["Rad"].attrs

    ds["Rad"].attrs = {}
    ds["Rad"].attrs = dict(
        long_name=attrs_rad["long_name"],
        standard_name=attrs_rad["standard_name"],
        units=attrs_rad["units"],
    )
    ds.attrs = {}
    
    return ds


def geoprocess_goes16_modis_match(
    modis_read_dir: str = "./",
    modis_satellite: str="aqua",
    goes_read_dir: str = "./",
    goes_save_path: str = "./",
    time_window: str="15 minutes",
    resolution: int=5_000

):
    logger.info(f"Starting Script")

    logger.info(f"Grabbing all MODIS files...")
    all_modis_files = get_list_filenames(modis_read_dir, ".nc")
    # filter files for terra files ONLY ( our satellite id)
    all_modis_files = list(filter(lambda x: modis_satellite in x, all_modis_files))
    logger.debug(f"Number of MODIS Files found: {len(all_modis_files)}")

    # grab MODIS datetimes
    logger.info(f"Getting unique times from files...")
    goes16_dateparser = lambda x: Path(x).name.split("_")[0]
    unique_modis_times = list(set(map(goes16_dateparser, all_modis_files)))
    logger.debug(f"Number of unique times found: {len(unique_modis_times)}")
    
    logger.info(f"Grabbing all GOES files...")
    all_goes16_files = get_list_filenames(goes_read_dir, ".nc")
    all_modis_files = list(filter(lambda x: modis_satellite in x, all_goes16_files))
    
    logger.debug(f"Number of MODIS Files found: {len(all_modis_files)}")
    logger.info(f"Selecting Nearest time...")
    logger.info(f"Check of time in window...")


if __name__ == '__main__':
    """
    # =========================
    # Test Cases
    # =========================

    # =========================
    # FAILURE TEST CASES
    # =========================
    """
    typer.run(geoprocess_goes16_modis_match)
