import autoroot
import os
import numpy as np
import rioxarray
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from tqdm import tqdm
from rs_tools._src.utils.io import get_list_filenames
import typer
import pygrib
from loguru import logger
import xarray as xr
import datetime
from satpy import Scene
from rs_tools._src.geoprocessing.interp import resample_rioxarray
from rs_tools._src.geoprocessing.msg.reproject import add_msg_crs
from rs_tools._src.geoprocessing.reproject import convert_lat_lon_to_x_y, calc_latlon
from rs_tools._src.geoprocessing.utils import check_sat_FOV
from rs_tools._src.geoprocessing.match import match_timestamps
from rs_tools._src.geoprocessing.msg import MSG_WAVELENGTHS
import pandas as pd
from datetime import datetime 
from functools import partial
import dask
import warnings

dask.config.set(**{'array.slicing.split_large_chunks': False})
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from datetime import datetime
from pathlib import Path

# TODO: Add unit conversion?

def parse_msg_dates_from_file(file: str):
    """
    Parses the date and time information from a MSG file name.

    Args:
        file (str): The file name to parse.

    Returns:
        str: The parsed date and time in the format 'YYYYJJJHHMM'.
    """
    timestamp = Path(file).name.split("-")[-2]
    timestamp = timestamp.split(".")[0]
    return timestamp


@dataclass
class MSGGeoProcessing:
    """
    A class for geoprocessing MSG data.

    Attributes:
        resolution (float): The resolution in meters.
        read_path (str): The path to read the files from.
        save_path (str): The path to save the processed files to.
        region (Tuple[str]): The region of interest defined by the bounding box coordinates (lon_min, lat_min, lon_max, lat_max).
        resample_method (str): The resampling method to use.

    Methods:
        msg_files(self) -> List[str]: Returns a list of all MSG files in the read path.
        preprocess_fn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]: Preprocesses the input dataset by applying corrections, subsetting, and resampling.
        preprocess_fn_radiances(self, ds: xr.Dataset) -> xr.Dataset: Preprocesses the input dataset for radiances.
        preprocess_fn_cloudmask(self, ds: xr.Dataset) -> xr.Dataset: Preprocesses the input dataset for cloud mask.
        preprocess_files(self): Preprocesses the files in the read path and saves the processed files to the save path.
        preprocess_radiances(self, files: List[str]) -> xr.Dataset: Preprocesses the radiances from the input files.
        preprocess_cloud_mask(self, files: List[str]) -> xr.Dataset: Preprocesses the cloud mask from the input files.
    """
    resolution: float
    read_path: str
    save_path: str
    region: Optional[Tuple[int, int, int, int]]
    resample_method: str

    @property
    def msg_files(self) -> List[str]:
        """
        Returns a list of all MSG files in the read path.

        Returns:
            List[str]: A list of file paths.
        """
        # get a list of all MSG radiance files from specified path
        files_radiances = get_list_filenames(self.read_path, ".nat")
        # get a list of all MSG cloud mask files from specified path
        files_cloudmask = get_list_filenames(self.read_path, ".grb")
        return files_radiances, files_cloudmask

    def preprocess_fn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Preprocesses the input dataset by applying corrections, subsetting, and resampling etc.

        Args:
            ds (xr.Dataset): The input dataset.

        Returns:
            Tuple[xr.Dataset, xr.Dataset]: The preprocessed dataset and the original dataset.
        """
        # copy to avoid modifying original dataset
        ds = ds.copy() 

        # assign coordinate reference system
        ds = add_msg_crs(ds)

        if self.region != (None, None, None, None):
            logger.info(f"Subsetting data to region: {self.region}")
            # subset data
            lon_bnds = (self.region[0], self.region[2])
            lat_bnds = (self.region[1], self.region[3])
            # convert lat lon bounds to x y (in meters)
            x_bnds, y_bnds = convert_lat_lon_to_x_y(ds.rio.crs, lon=lon_bnds, lat=lat_bnds, )
            # check that region is within the satellite field of view
            # compile satellite FOV
            satellite_FOV = (min(ds.x.values), min(ds.y.values), max(ds.x.values), max(ds.y.values))
            # compile region bounds in x y
            region_xy = (x_bnds[0], y_bnds[0], x_bnds[1], y_bnds[1])
            if not check_sat_FOV(region_xy, FOV=satellite_FOV):
                raise ValueError("Region is not within the satellite field of view")

            ds = ds.sortby("x").sortby("y")
            # slice based on x y bounds
            ds_subset = ds.sel(y=slice(y_bnds[0], y_bnds[1]), x=slice(x_bnds[0], x_bnds[1]))
        else:
            ds_subset = ds

        if self.resolution is not None:
            logger.info(f"Resampling data to resolution: {self.resolution} m")
            # resampling
            ds_subset = resample_rioxarray(ds_subset, resolution=(self.resolution, self.resolution), method=self.resample_method)

        # assign coordinates
        ds_subset = calc_latlon(ds_subset)

        return ds_subset, ds

    def preprocess_fn_radiances(self, file: List[str], cloud_mask: np.array) -> xr.Dataset:
        """
        Preprocesses the MSG radiance dataset.

        Args:
            file (List[str]): The input file.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """

        # Load file using satpy scenes
        scn = Scene(
            reader="seviri_l1b_native",
            filenames=file
        )
        # Load radiance bands
        channels = [x for x in scn.available_dataset_names() if x!='HRV']
        assert len(channels) == 11, "Number of channels is not 11"

        scn.load(channels, generate=False, calibration='radiance')
        
        # change to xarray data
        ds = scn.to_xarray()

        # attach cloud mask as data variable before preprocessing
        ds = ds.assign(cloud_mask=(("y", "x"), cloud_mask))

        # reset coordinates for resampling/reprojecting
        # this drops all {channel}_acq_time coordinates
        ds = ds.reset_coords(drop=True)

        # do core preprocess function (e.g. resample, add crs etc.)
        ds_subset, ds = self.preprocess_fn(ds) 

        # Store the attributes in a dict before concatenation
        attrs_dict = {x: ds_subset[x].attrs for x in channels}
            
        # concatinate in new band dimension
        # NOTE: Concatination overwrites attrs of bands.
        ds_subset = ds_subset.assign(Rad=xr.concat(list(map(lambda x: ds_subset[x], channels)), dim="band"))
        # rename band dimensions
        ds_subset = ds_subset.assign_coords(band=list(map(lambda x: x, channels)))

        # re-index coordinates
        ds_subset = ds_subset.set_coords(['latitude', 'longitude', 'cloud_mask'])

        # drop variables that will no longer be needed
        ds_subset = ds_subset.drop(list(map(lambda x: x, channels)))

        # extract measurement time
        time_stamp = attrs_dict[list(attrs_dict.keys())[0]]['start_time']
        # assign bands and time data to each variable
        ds_subset = ds_subset.expand_dims({"time": [time_stamp]})

        # NOTE: Keep only certain relevant attributes
        ds_subset.attrs = {}
        ds_subset.attrs = dict(
            calibration=attrs_dict[list(attrs_dict.keys())[0]]["calibration"],
            standard_name=attrs_dict[list(attrs_dict.keys())[0]]["standard_name"],
            platform_name=attrs_dict[list(attrs_dict.keys())[0]]["platform_name"],
            sensor=attrs_dict[list(attrs_dict.keys())[0]]["sensor"],
            units=attrs_dict[list(attrs_dict.keys())[0]]["units"],
            orbital_parameters=attrs_dict[list(attrs_dict.keys())[0]]["orbital_parameters"]
        )
        
        # TODO: Correct wavelength assignment. This attaches 36++ wavelengths to each band.
        # assign band wavelengths 
        ds_subset = ds_subset.expand_dims({"band_wavelength": list(MSG_WAVELENGTHS.values())}) 

        return ds_subset

    def preprocess_fn_cloudmask(self, file: List[str]) -> np.array:
        """
        Preprocesses the input dataset for MSG cloud masks.

        Args:
            file (List[str]): The input file.

        Returns:
            np.array: The preprocessed cloud mask dataset.
        """

        grbs = pygrib.open(file[0])
        # Loop over all messages in the GRIB file
        for grb in grbs:
            if grb.name == 'Cloud mask':
                # Extract values from grb and return np.array
                cloud_mask = grb.values
                return cloud_mask

    def preprocess_radiances(self, files: List[str], cloud_mask: np.array) -> xr.Dataset:
        """
        Preprocesses radiances from the input files.

        Args:
            files (List[str]): The list of file paths.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        # Check that all files contain radiance data
        file = list(filter(lambda x: ".nat" in x, files))

        # Check that only one file is selected
        logger.info(f"Number of radiance files: {len(file)}")
        assert len(file) == 1

        # load file using satpy, convert to xarray dataset, and preprocess
        ds = self.preprocess_fn_radiances(file, cloud_mask=cloud_mask)

        return ds

    def preprocess_cloud_mask(self, files: List[str]) -> xr.Dataset:
        """
        Preprocesses cloud mask from the input files.

        Args:
            files (List[str]): The list of file paths.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        # Check that all files contain cloud mask data
        file = list(filter(lambda x: "CLMK" in x, files))
        
        # Check that only one file is present
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
        files_radiances, files_cloudmask = self.msg_files
        unique_times_radiances = list(set(map(parse_msg_dates_from_file, files_radiances)))
        unique_times_cloudmask = list(set(map(parse_msg_dates_from_file, files_cloudmask)))

        df_matches = match_timestamps(unique_times_radiances, unique_times_cloudmask, cutoff=15) 

        pbar_time = tqdm(df_matches["timestamps_data"].values)

        for itime in pbar_time:

            pbar_time.set_description(f"Processing: {itime}")

            # get cloud mask file for specific time
            itime_cloud = df_matches.loc[df_matches["timestamps_data"] == itime, "timestamps_cloudmask"].values[0]
            files_cloud = list(filter(lambda x: itime_cloud in x, files_cloudmask))

            try:
                # load cloud mask
                cloud_mask = self.preprocess_cloud_mask(files_cloud)
            except AssertionError:
                logger.error(f"Skipping {itime} due to missing cloud mask")
                continue

            # get data files for specific time
            files = list(filter(lambda x: itime in x, files_radiances))

            try:
                # load radiances and attach cloud mask
                ds = self.preprocess_radiances(files, cloud_mask=cloud_mask)
            except AssertionError:
                logger.error(f"Skipping {itime} due to error loading")
                continue

             # remove crs from dataset
            ds = ds.drop_vars('msg_seviri_fes_3km') 

            # remove attrs that cause netcdf error
            for var in ds.data_vars:
                ds[var].attrs.pop('grid_mapping', None)

            # check if save path exists, and create if not
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        
            # remove file if it already exists
            save_filename = Path(self.save_path).joinpath(f"{itime}_msg.nc")
            if os.path.exists(save_filename):
                logger.info(f"File already exists. Overwriting file: {save_filename}")
                os.remove(save_filename)

            # save to netcdf
            ds.to_netcdf(save_filename, engine="netcdf4")

def geoprocess_msg(
        resolution: float = None, # defined in meters
        read_path: str = "./",
        save_path: str = "./",
        region: Tuple[int, int, int, int] = (None, None, None, None),
        resample_method: str = "bilinear",
):
    """
    Geoprocesses MSG files

    Args:
        resolution (float, optional): The resolution in meters to resample data to. Defaults to None.
        read_path (str, optional): The path to read the files from. Defaults to "./".
        save_path (str, optional): The path to save the geoprocessed files to. Defaults to "./".
        region (Tuple[int, int, int, int], optional): The geographic region to extract (lon_min, lat_min, lon_max, lat_max). Defaults to None.
        resample_method (str, optional): The resampling method to use. Defaults to "bilinear".

    Returns:
        None
    """
    # Initialize MSG GeoProcessor
    logger.info(f"Initializing MSG GeoProcessor...")
    msg_geoprocessor = MSGGeoProcessing(
        resolution=resolution, 
        read_path=read_path, 
        save_path=save_path,
        region=region,
        resample_method=resample_method
        )
    logger.info(f"GeoProcessing Files...")
    msg_geoprocessor.preprocess_files()

    logger.info(f"Finished MSG GeoProcessing Script...!")


if __name__ == '__main__':
    """
    # =========================
    # Test Cases
    # =========================
    python geoprocessor_msg.py --read-path "/home/data" --save-path /home/data/msg/geoprocessed
    python geoprocessor_msg.py --read-path "/home/data" --save-path /home/data/msg/geoprocessed --resolution 5000
    python geoprocessor_msg.py --read-path "/home/data" --save-path /home/data/msg/geoprocessed --resolution 10000
    python geoprocessor_msg.py --read-path "/home/data" --save-path /home/data/msg/geoprocessed --region (-10, -10, 10, 10)
    python geoprocessor_msg.py --read-path "/home/data" --save-path /home/data/msg/geoprocessed --resolution 2000 --region (-10, -10, 10, 10)
            
    # =========================
    # FAILURE TEST CASES
    # =========================
    python geoprocessor_msg.py --read-path "/home/data" --save-path /home/data/msg/geoprocessed --resolution 2000 --region (-100, -10, -90, 10)
    """
    typer.run(geoprocess_msg)
