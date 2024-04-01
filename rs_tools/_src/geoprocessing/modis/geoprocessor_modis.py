import autoroot
import os
import numpy as np
from xrpatcher._src.base import XRDAPatcher
import rioxarray
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from tqdm import tqdm
from rs_tools import modis_download, MODIS_VARIABLES, get_modis_channel_numbers
from rs_tools._src.geoprocessing.interp import resample_rioxarray
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.grid import create_latlon_grid
import typer
from loguru import logger
import xarray as xr
from satpy import Scene
import datetime
from rs_tools._src.data.modis import MODISFileName, MODIS_ID_TO_NAME, MODIS_NAME_TO_ID, get_modis_paired_files
from rs_tools._src.geoprocessing.modis.reproject import add_modis_crs
from rs_tools._src.geoprocessing.modis import MODIS_WAVELENGTHS, parse_modis_dates_from_file, format_modis_dates
import pandas as pd
from datetime import datetime
from pathlib import Path
import dask
import warnings

dask.config.set(**{'array.slicing.split_large_chunks': False})
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class MODISGeoProcessing:
    satellite: str
    read_path: str
    save_path: str
    """
    A class for geoprocessing MODIS data.

    Attributes:
        satellite (str): The satellite to geoprocess data for.
        read_path (str): The path to read the files from.
        save_path (str): The path to save the processed files to.

    Methods:
        modis_files(self) -> List[str]: Returns a list of all MODIS files in the read path.
        preprocess_fn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]: Preprocesses the input dataset by applying corrections.
        preprocess_fn_radiances(self, ds: xr.Dataset) -> xr.Dataset: Preprocesses the input dataset for radiances.
        preprocess_fn_cloudmask(self, ds: xr.Dataset) -> xr.Dataset: Preprocesses the input dataset for cloud mask.
        preprocess_files(self): Preprocesses the files in the read path and saves the processed files to the save path.
        preprocess_radiances(self, files: List[str]) -> xr.Dataset: Preprocesses the radiances from the input files.
        preprocess_cloud_mask(self, files: List[str]) -> xr.Dataset: Preprocesses the cloud mask from the input files.
    """
    @property
    def modis_files(self) -> List[str]:
        """
        Returns a list of all MODIS files in the read path.

        Returns:
            List[str]: A list of file paths.
        """
        # get a list of all MODIS filenames within the path
        files = get_list_filenames(self.read_path, ".hdf")
        return files
    
    def preprocess_fn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Preprocesses the input dataset by applying corrections etc.

        Args:
            ds (xr.Dataset): The input dataset.

        Returns:
            ds (xr.Dataset):: The preprocessed dataset.
        """
        # copy to avoid modifying original dataset
        ds = ds.copy() 

        # assign coordinate reference system
        ds = add_modis_crs(ds)

        # TODO: Add functionality to resample data to specific resolution

        return ds

    def preprocess_fn_radiances(self, file: List[str]) -> xr.Dataset:
        """
        Preprocesses the MODIS radiance dataset.

        Args:
            file (List[str]): The input file.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        # Load file using satpy scenes
        scn = Scene(
            reader="modis_l1b",
            filenames=file
        )
        # Load radiance bands
        channels = get_modis_channel_numbers()
        scn.load(channels, generate=False, calibration='radiance')
        
        # change to xarray data
        ds = scn.to_xarray_dataset()  

        # do core preprocess function (e.g. resample, add crs etc.)
        ds = self.preprocess_fn(ds) 

        # Store the attributes in a dict before concatenation
        attrs_dict = {x: ds[x].attrs for x in channels}
            
        # concatinate in new band dimension
        # NOTE: Concatination overwrites attrs of bands.
        ds = ds.assign(Rad=xr.concat(list(map(lambda x: ds[x], channels)), dim="band"))
        # rename band dimensions
        ds = ds.assign_coords(band=list(map(lambda x: x, channels)))

        # convert measurement time (in seconds) to datetime
        time_stamp = pd.to_datetime(ds.attrs['start_time'])
        time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")  
        # assign bands and time data to each variable
        ds = ds.expand_dims({"time": [time_stamp]})

        # NOTE: Keep only certain relevant attributes
        ds.attrs = {}
        ds.attrs = dict(
            calibration=attrs_dict[list(attrs_dict.keys())[0]]["calibration"],
            standard_name=attrs_dict[list(attrs_dict.keys())[0]]["standard_name"],
            platform_name=attrs_dict[list(attrs_dict.keys())[0]]["platform_name"],
            sensor=attrs_dict[list(attrs_dict.keys())[0]]["sensor"],
            units=attrs_dict[list(attrs_dict.keys())[0]]["units"],
        )
        
        # TODO: Correct wavelength assignment. This attaches 36++ wavelengths to each band.
        # assign band wavelengths 
        ds = ds.expand_dims({"band_wavelength": list(MODIS_WAVELENGTHS.values())})   

        return ds

    def preprocess_fn_cloudmask(self, file: List[str]) -> xr.Dataset:
        """
        Preprocesses the input dataset for MODIS cloud masks.

        Args:
            file (List[str]): The input file.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """    
        # Load file using satpy scenes
        scn = Scene(
            reader="modis_l2",
            filenames=file
        )
        # Load cloud mask data
        datasets = scn.available_dataset_names()
        # Needs to be loaded at 1000 m resolution for all channels to match
        scn.load(datasets, generate=False, resolution=1000) 
        
        # change to xarray data
        ds = scn.to_xarray_dataset()

        return ds

    def preprocess_radiances(self, files: List[str]) -> xr.Dataset:
        """
        Preprocesses radiances from the input files.

        Args:
            files (List[str]): The list of file paths.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        identifier = MODIS_NAME_TO_ID[self.satellite]
        
        # Check that all files contain radiance data
        file = list(filter(lambda x: identifier in x, files))

        # Check that only one file is selected
        logger.info(f"Number of radiance files: {len(file)}")
        assert len(file) == 1

        # load file using satpy, convert to xarray dataset, and preprocess
        ds = self.preprocess_fn_radiances(file)
        
        return ds


    def preprocess_cloud_mask(self, files: List[str]) -> xr.Dataset:
        """
        Preprocesses cloud mask from the input files.

        Args:
            files (List[str]): The list of file paths.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        identifier = MODIS_NAME_TO_ID[f'{self.satellite}_cloud']
        
        # Check that all files contain radiance data
        file = list(filter(lambda x: identifier in x, files))

        # Check that only one file is selected
        logger.info(f"Number of cloud mask files: {len(file)}")
        assert len(file) == 1

        # load file using satpy, convert to xarray dataset, and preprocess
        ds = self.preprocess_fn_cloudmask(file)

        return ds


    def preprocess_files(self):
        """
        Preprocesses multiple files in read path and saves processed files to save path.
        """
        # get unique times from read path
        unique_times = list(set(map(parse_modis_dates_from_file, self.modis_files)))

        pbar_time = tqdm(unique_times)

        for itime in pbar_time:

            pbar_time.set_description(f"Processing: {itime}")

            # get files from unique times
            files = list(filter(lambda x: itime in x, self.modis_files))

            try:
                # load radiances
                ds = self.preprocess_radiances(files)
            except AssertionError:
                logger.error(f"Skipping {itime} due to error loading")
                continue
            try:
                # load cloud mask
                ds_clouds = self.preprocess_cloud_mask(files)["cloud_mask"]
            except AssertionError:
                logger.error(f"Skipping {itime} due to missing cloud mask")
                continue

            # save cloud mask as data coordinate
            ds = ds.assign_coords({"cloud_mask": (("y", "x"), ds_clouds.values)})
            # add cloud mask attrs to dataset
            ds["cloud_mask"].attrs = ds_clouds.attrs

            # remove attrs that cause netcdf error
            for attr in ["start_time", "end_time", "area", "_satpy_id"]:
                ds["cloud_mask"].attrs.pop(attr)

            for var in ds.data_vars:
                ds[var].attrs.pop('start_time', None)
                ds[var].attrs.pop('end_time', None)
                ds[var].attrs.pop('area', None)
                ds[var].attrs.pop('_satpy_id', None)

            # remove crs from dataset
            ds = ds.drop_vars("crs")

            # check if save path exists, and create if not
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        
            # remove file if it already exists
            itime_name = format_modis_dates(itime)
            save_filename = Path(self.save_path).joinpath(f"{itime_name}_{self.satellite}.nc")
            if os.path.exists(save_filename):
                logger.info(f"File already exists. Overwriting file: {save_filename}")
                os.remove(save_filename)

            # save to netcdf
            ds.to_netcdf(save_filename, engine="netcdf4")

def geoprocess_modis(
        satellite: str = "terra",
        read_path: str = "/Users/anna.jungbluth/Desktop/git/rs_tools/data/terra",
        save_path: str = "/Users/anna.jungbluth/Desktop/git/rs_tools/data/terra/geoprocessed"
):
    """
    Geoprocesses MODIS files

    Args:
        satellite (str, optional): The satellite to download files for.
        read_path (str, optional): The path to read the files from. Defaults to "./".
        save_path (str, optional): The path to save the downloaded files. Defaults to "./".

    Returns:
        None
    """
    # Initialize MODIS GeoProcessor
    logger.info(f"Initializing {satellite.upper()} GeoProcessor...")

    modis_geoprocessor = MODISGeoProcessing(
        satellite=satellite, 
        read_path=read_path, 
        save_path=save_path
        )
    logger.info(f"GeoProcessing Files...")
    modis_geoprocessor.preprocess_files()
    
    logger.info(f"Finished {satellite.upper()} GeoProcessing Script...!")


if __name__ == '__main__':
    """
    # =========================
    # Test Cases
    # =========================
    python geoprocessor_modis.py --satellite aqua --read-path "/home/data" --save-path /home/data/modis/geoprocessed
    python geoprocessor_modis.py --satellite terra --read-path "/home/data" --save-path /home/data/modis/geoprocessed
    
    # ====================
    # FAILURE TEST CASES
    # ====================
    """
    typer.run(geoprocess_modis)
