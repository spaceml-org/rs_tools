import autoroot
import typer
from pathlib import Path
from loguru import logger
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyproj import CRS
from goes2go import GOES
from datetime import datetime
from rs_tools._src.geoprocessing.goes.validation import correct_goes16_bands, correct_goes16_satheight
from rs_tools._src.geoprocessing.goes.reproject import add_goes16_crs
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.modis import parse_modis_dates_from_file
from rs_tools._src.data.modis import MODIS_NAME_TO_ID


BANDS_GOES16 = list(np.arange(1, 17))


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


def download_goes16_modis_match(
    modis_read_dir: str = "./",
    modis_satellite: str="aqua",
    goes_save_dir: str = "./",
    time_window: str="15 minutes",

):
    logger.info(f"Starting Script")

    logger.info(f"Grabbing all MODIS files...")
    all_modis_files = get_list_filenames(modis_read_dir, ".nc")
    # filter files for terra files ONLY ( our satellite id)
    all_modis_files = list(filter(lambda x: modis_satellite in x, all_modis_files))
    logger.debug(f"Number of MODIS Files found: {len(all_modis_files)}")

    # grab MODIS datetimes
    logger.info(f"Getting unique times from files...")
    unique_modis_times = list(set(map(lambda x: Path(x).name.split("_")[0], all_modis_files)))
    logger.debug(f"Number of unique times found: {len(unique_modis_times)}")
    print(unique_modis_times[0])
    
    logger.info(f"Grabbing all GOES16 files")
    logger.info(f"Selecting Nearest time window...")

    
    time_unit, time_freq = time_window.split(" ")
    within = pd.to_timedelta(float(time_unit), unit=time_freq)

    # ABI Level 1b Data
    G_L1b = GOES(
        satellite=16, 
        product="ABI-L1b-Rad",
        domain='F',
        channel=BANDS_GOES16
    )

    # ABI Level 2 Cloud Masks Data
    G_CM = GOES(
        satellite=16,
        product="ABI-L2-ACM",
        domain='F',
    )
    logger.info(f"Starting loop...")
    pbar = tqdm(unique_modis_times)
    
    for itime in pbar:

        time_stamp = datetime.strptime(itime, "%Y%m%d%H%M%S")

        pbar.set_description(f"Download Nearest GOES Time Stamp: {time_stamp}")

        G_L1b.nearesttime(
            attime=time_stamp,
            within=within,
            return_as="filelist",
            save_dir=goes_save_dir
        )
        G_CM.nearesttime(
            attime=time_stamp,
            within=within,
            return_as="filelist",
            save_dir=goes_save_dir
        )
    logger.info(f"Finished script...!")


if __name__ == '__main__':
    """
    # =========================
    # Test Cases
    # =========================

    # =========================
    # FAILURE TEST CASES
    # =========================
    """
    typer.run(download_goes16_modis_match)