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
from loguru import logger
import xarray as xr
import datetime
from rs_tools._src.geoprocessing.interp import resample_rioxarray
from rs_tools._src.geoprocessing.goes.validation import correct_goes16_bands, correct_goes16_satheight
from rs_tools._src.geoprocessing.goes.reproject import add_goes16_crs
from rs_tools._src.geoprocessing.reproject import convert_lat_lon_to_x_y, calc_latlon
from rs_tools._src.geoprocessing.utils import check_sat_FOV
import pandas as pd
from datetime import datetime
from functools import partial
import dask
import warnings

dask.config.set(**{'array.slicing.split_large_chunks': False})
warnings.filterwarnings('ignore', category=FutureWarning)

def parse_goes16_dates_from_file(file: str):
    timestamp = Path(file).name.replace("-","_").split("_")
    return datetime.strptime(timestamp[-2][1:], "%Y%j%H%M%S%f").strftime("%Y%j%H%M")


@dataclass
class GOES16GeoProcessing:
    """
    A class for performing geoprocessing on GOES-16 satellite data.

    Attributes:
        resolution (float): The resolution in meters.
        read_path (str): The path to read the files from.
        save_path (str): The path to save the processed files to.
        region (Tuple[str]): The region of interest defined by the bounding box coordinates (lon_min, lat_min, lon_max, lat_max).
        resample_method (str): The resampling method to use.

    Methods:
        goes_files(self) -> List[str]: Returns a list of all GOES files in the read path.
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
    def goes_files(self) -> List[str]:
        """
        Returns a list of all GOES files in the read path.

        Returns:
            List[str]: A list of file paths.
        """
        # get a list of all GOES files from specified path
        files = get_list_filenames(self.read_path, ".nc")
        return files

    def preprocess_fn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Preprocesses the input dataset by applying corrections, subsetting, and resampling.

        Args:
            ds (xr.Dataset): The input dataset.

        Returns:
            Tuple[xr.Dataset, xr.Dataset]: The preprocessed dataset and the original dataset.
        """
        # copy to avoid modifying original dataset
        ds = ds.copy() 

        # convert measurement angles to horizontal distance in meters
        ds = correct_goes16_satheight(ds) 
        try:
            # correct band coordinates to reorganize xarray dataset
            ds = correct_goes16_bands(ds) 
        except AttributeError:
            pass
        # assign coordinate reference system
        ds = add_goes16_crs(ds)

        if self.region != (None, None, None, None):
            logger.info(f"Subsetting data to region: {self.region}")
            # subset data
            lon_bnds = (self.region[0], self.region[2])
            lat_bnds = (self.region[1], self.region[3])
            # convert lat lon bounds to x y (in meters)
            x_bnds, y_bnds = convert_lat_lon_to_x_y(ds.FOV.crs, lon=lon_bnds, lat=lat_bnds, )
            # check that region is within the satellite field of view
            # compile satellite FOV
            satellite_FOV = (min(ds.x.values), min(ds.y.values), max(ds.x.values), max(ds.y.values))
            # compile region bounds in x y
            region_xy = (x_bnds[0], y_bnds[0], x_bnds[1], y_bnds[1])
            if not check_sat_FOV(region_xy, FOV=satellite_FOV):
                raise ValueError("Region is not within the satellite field of view")

            ds = ds.sortby("x").sortby("y")
            # slice based on x y bounds
            ds = ds.sel(y=slice(y_bnds[0], y_bnds[1]), x=slice(x_bnds[0], x_bnds[1]))

        if self.resolution is not None:
            logger.info(f"Resampling data to resolution: {self.resolution} m")
            # resampling
            ds_subset = resample_rioxarray(ds, resolution=self.resolution, method=self.resample_method)
        else:
            ds_subset = ds

        # assign coordinates
        ds_subset = calc_latlon(ds_subset)

        return ds_subset, ds

    def preprocess_fn_radiances(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Preprocesses the input dataset for GOES16 radiances.

        Args:
            ds (xr.Dataset): The input dataset.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        variables = ["Rad", "DQF"] # "Rad" = radiance, "DQF" = data quality flag

        # do core preprocess function (e.g. to correct band coordinates, subset data, resample, etc.)
        ds_subset, ds = self.preprocess_fn(ds)

        # select relevant variables
        ds_subset = ds_subset[variables]
        # convert measurement time (in seconds) to datetime
        time_stamp = pd.to_datetime(ds.t.values) 
        time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M") 
        # assign bands and time data to each variable
        ds_subset[variables] = ds_subset[variables].expand_dims({"band": ds.band.values, "time": [time_stamp]})
        # drop variables that will no longer be needed
        ds_subset = ds_subset.drop_vars(["t", "y_image", "x_image", "goes_imager_projection"])
        # assign band attributes to dataset
        ds_subset.band.attrs = ds.band.attrs
        # TODO: Correct wavelength assignment. This attaches 16 wavelengths to each band.
        # assign band wavelength to each variable
        ds_subset = ds_subset[variables].expand_dims({"band_wavelength": ds.band_wavelength.values})
        ds_subset.band_wavelength.attrs = ds.band_wavelength.attrs

        return ds_subset

    def preprocess_fn_cloudmask(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Preprocesses the input dataset for GOES16 cloud masks.

        Args:
            ds (xr.Dataset): The input dataset.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        variables = ["BCM"]

        # do core preprocess function
        ds_subset, ds = self.preprocess_fn(ds)

        # select relevant variable
        ds_subset = ds_subset[variables]
        # convert measurement time (in seconds) to datetime
        time_stamp = pd.to_datetime(ds.t.values)
        time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")
        # assign time data to variable
        ds_subset = ds_subset.expand_dims({"time": [time_stamp]})
        # drop variables that will no longer be needed
        ds_subset = ds_subset.drop_vars(["t", "y_image", "x_image", "goes_imager_projection"])

        return ds_subset

    def preprocess_radiances(self, files: List[str]) -> xr.Dataset:
        """
        Preprocesses radiances from the input files.

        Args:
            files (List[str]): The list of file paths.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        # Check that all files contain radiance data
        files = list(filter(lambda x: "Rad" in x, files))

        # Check that all 16 bands are present
        logger.info(f"Number of radiance files: {len(files)}")
        assert len(files) == 16

        # open multiple files as a single dataset
        ds = [xr.open_mfdataset(ifile, preprocess=self.preprocess_fn_radiances, concat_dim="band", combine="nested") for
              ifile in files]
        # reinterpolate to match coordinates of the first image
        ds = [ds[0]] + [ids.interp(x=ds[0].x, y=ds[0].y) for ids in ds[1:]]
        # concatenate in new band dimension
        ds = xr.concat(ds, dim="band")

        # NOTE: Keep only certain relevant attributes
        attrs_rad = ds["Rad"].attrs

        ds["Rad"].attrs = {}
        ds["Rad"].attrs = dict(
            long_name=attrs_rad["long_name"],
            standard_name=attrs_rad["standard_name"],
            units=attrs_rad["units"],
        )
        ds["DQF"].attrs = {}

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
        files = list(filter(lambda x: "ACMF" in x, files))
        
        # Check that only one file is present
        logger.info(f"Number of cloud mask files: {len(files)}")
        assert len(files) == 1

        # open multiple files as a single dataset
        ds = xr.open_mfdataset(files[0])
        ds = self.preprocess_fn_cloudmask(ds)

        # NOTE: Keep only certain relevant attributes
        attrs_bcm = ds["BCM"].attrs
        ds = ds.rename({"BCM": "cloud_mask"})
        ds["cloud_mask"].attrs = {}
        ds["cloud_mask"].attrs = dict(
            long_name=attrs_bcm["long_name"],
            standard_name=attrs_bcm["standard_name"],
            units=attrs_bcm["units"],
        )

        return ds

    def preprocess_files(self):
        """
        Preprocesses multiple files in the read path and saves processed files to the save path.
        """
        # get unique times from read path
        unique_times = list(set(map(parse_goes16_dates_from_file, self.goes_files)))

        pbar_time = tqdm(unique_times)

        for itime in pbar_time:

            pbar_time.set_description(f"Processing: {itime}")

            # get files from unique times
            files = list(filter(lambda x: itime in x, self.goes_files))

            try:
                # load radiances
                ds = self.preprocess_radiances(files)
            except AssertionError:
                logger.error(f"Skipping {itime} due to missing bands")
                continue
            try:
                # load cloud mask
                ds_clouds = self.preprocess_cloud_mask(files)["cloud_mask"]
            except AssertionError:
                logger.error(f"Skipping {itime} due to missing cloud mask")
                continue

            # interpolate cloud mask to data
            ds_clouds = ds_clouds.interp(x=ds.x, y=ds.y)
            # save cloud mask as data coordinate
            ds = ds.assign_coords({"cloud_mask": (("y", "x"), ds_clouds.values.squeeze())})
            ds["cloud_mask"].attrs = ds_clouds.attrs

            # check if save path exists, and create if not
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        
            # remove file if it already exists
            save_filename = Path(self.save_path).joinpath(f"{itime}_goes16.nc")
            if os.path.exists(save_filename):
                logger.info(f"File already exists. Overwriting file: {save_filename}")
                os.remove(save_filename)
            # save to netcdf
            ds.to_netcdf(save_filename, engine="netcdf4")


def geoprocess_goes16(
        resolution: float = None, # defined in meters
        read_path: str = "./",
        save_path: str = "./",
        region: Tuple[int, int, int, int] = (None, None, None, None),
        resample_method: str = "bilinear",
):
    """
    Downloads MODIS TERRA and GOES 16 files for the specified period, region, and save path.

    Args:
        resolution (float, optional): The resolution of the downloaded files in meters. Defaults to None.
        read_path (str, optional): The path to read the files from. Defaults to "./".
        save_path (str, optional): The path to save the downloaded files. Defaults to "./".
        region (Tuple[int, int, int, int], optional): The geographic region to download files for. Defaults to None.
        resample_method (str, optional): The resampling method to use. Defaults to "bilinear".

    Returns:
        None
    """
    # Initialize GOES 16 GeoProcessor
    logger.info(f"Initializing GOES16 GeoProcessor...")
    goes16_geoprocessor = GOES16GeoProcessing(
        resolution=resolution, 
        read_path=read_path, 
        save_path=save_path,
        region=region,
        resample_method=resample_method
        )
    logger.info(f"GeoProcessing Files...")
    goes16_geoprocessor.preprocess_files()

    logger.info(f"Finished GOES 16 GeoProcessing Script...!")


if __name__ == '__main__':
    """
    # =========================
    # Test Cases
    # =========================
    python geoprocessor_goes16.py --read-path "/home/data" --save-path /home/data/goes/geoprocessed
    python geoprocessor_goes16.py --read-path "/home/data" --save-path /home/data/goes/geoprocessed --resolution 1000
    python geoprocessor_goes16.py --read-path "/home/data" --save-path /home/data/goes/geoprocessed --resolution 5000
    python geoprocessor_goes16.py --read-path "/home/data" --save-path /home/data/goes/geoprocessed --resolution 5000 --region -130 -15 -90 5

    # ====================
    # FAILURE TEST CASES
    # ====================
    python geoprocessor_goes16.py --read-path "/home/data" --save-path /home/data/goes/geoprocessed --resolution 5000 --region -200 -15 90 5
    """
    typer.run(geoprocess_goes16)
