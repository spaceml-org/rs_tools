import autoroot
import os
import gc
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
from rs_tools._src.geoprocessing.interp import resample_rioxarray
from rs_tools._src.geoprocessing.goes import parse_goes16_dates_from_file, format_goes_dates
from rs_tools._src.geoprocessing.goes.validation import correct_goes16_bands, correct_goes16_satheight
from rs_tools._src.geoprocessing.goes.reproject import add_goes16_crs
from rs_tools._src.geoprocessing.reproject import convert_lat_lon_to_x_y, calc_latlon
from rs_tools._src.geoprocessing.utils import check_sat_FOV
from rs_tools._src.geoprocessing.goes import GOES_WAVELENGTHS, GOES_CHANNELS
import pandas as pd
import dask
import warnings

dask.config.set(**{'array.slicing.split_large_chunks': False})
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class GOES16GeoProcessing:
    """
    A class for geoprocessing GOES-16 data.

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

    def preprocess_fn(self, ds: xr.Dataset, calc_coords=False) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Preprocesses the input dataset by applying corrections, subsetting, and resampling etc.

        Args:
            ds (xr.Dataset): The input dataset.
            calc_coords (bool): Whether to calculate latitude and longitude coordinates. Defaults to False.

        Returns:
            Tuple[xr.Dataset, xr.Dataset]: The preprocessed dataset and the original dataset.
        """
        # convert measurement angles to horizontal distance in meters
        ds = correct_goes16_satheight(ds) 
        try:
            # correct band coordinates to reorganize xarray dataset
            ds = correct_goes16_bands(ds) 
        except:
            pass
        # assign coordinate reference system
        ds = add_goes16_crs(ds)

        if self.region is not None:
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
            ds_subset = ds.sel(y=slice(y_bnds[0], y_bnds[1]), x=slice(x_bnds[0], x_bnds[1]))
        else:
            ds_subset = ds

        if self.resolution is not None:
            # TODO: Test how resampling impacts cloud_mask
            logger.info(f"Resampling data to resolution: {self.resolution} m")
            # resampling
            ds_subset = resample_rioxarray(ds_subset, resolution=(self.resolution, self.resolution), method=self.resample_method)
        
        # assign coordinates
        if calc_coords:
            logger.info("Assigning latitude and longitude coordinates.")
            ds_subset = calc_latlon(ds_subset)
        del ds # delete to avoid memory problems
        return ds_subset

    def preprocess_fn_radiances(self, ds: xr.Dataset, calc_coords=False) -> xr.Dataset:
        """
        Preprocesses the GOES16 radiance dataset.

        Args:
            ds (xr.Dataset): The input dataset.
            calc_coords (bool): Whether to calculate latitude and longitude coordinates. Defaults to False.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        variables = ["Rad"] # "Rad" = radiance, "DQF" = data quality flag

        # Extract relevant attributes from original dataset
        time_stamp = pd.to_datetime(ds.t.values) 
        band_attributes = ds.band.attrs
        band_wavelength_attributes = ds.band_wavelength.attrs
        band_wavelength_values = ds.band_wavelength.values
        band_values = ds.band.values
        # # Convert keys in dictionary to strings for easier comparison
        # GOES_CHANNELS_STR = {f'{round(key, 1)}': value for key, value in GOES_CHANNELS.items()}
        # # Round value and extract channel number
        # band_values = int(GOES_CHANNELS_STR[f'{round(band_wavelength_values[0], 1):.1f}'])

        # do core preprocess function (e.g. to correct band coordinates, subset data, resample, etc.)
        ds_subset = self.preprocess_fn(ds, calc_coords=calc_coords)

        # convert measurement time (in seconds) to datetime
        time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M") 
        # assign bands data to each variable
        ds_subset = ds_subset[variables]
        ds_subset = ds_subset.expand_dims({"band": band_values})
        # attach time coordinate
        ds_subset = ds_subset.assign_coords({"time": [time_stamp]})
        # drop variables that will no longer be needed
        # ds_subset = ds_subset.drop_vars(["t", "y_image", "x_image", "goes_imager_projection"])
        ds_subset = ds_subset.drop_vars(["t", "y_image", "x_image"]) # NOTE: Keep goes_imager_projection for easier access to crs
        # assign band attributes to dataset
        ds_subset.band.attrs = band_attributes
        # assign band wavelength to each variable
        ds_subset = ds_subset.assign_coords({"band_wavelength": band_wavelength_values})
        ds_subset.band_wavelength.attrs = band_wavelength_attributes
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

        time_stamp = pd.to_datetime(ds.t.values)

        # do core preprocess function
        ds_subset = self.preprocess_fn(ds)

        # select relevant variable
        ds_subset = ds_subset[variables]
        # convert measurement time (in seconds) to datetime
        time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")
        # assign time data to variable
        ds_subset = ds_subset.assign_coords({"time": [time_stamp]})
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

        for i, ifile in enumerate(files):
            with xr.load_dataset(ifile, engine='h5netcdf') as ds_file:
                logger.info(f"Loading file {i}/{len(files)}: {ifile}")
                if i == 0: 
                    # Preprocess first file and assign lat lon coords
                    ds_file = self.preprocess_fn_radiances(ds_file, calc_coords=True)
                    ds = ds_file
                else:
                    # Preprocess other files without calculating lat lon coords
                    ds_file = self.preprocess_fn_radiances(ds_file, calc_coords=False)
                    # reinterpolate to match coordinates of the first image
                    ds_file = ds_file.interp(x=ds.x, y=ds.y)
                    # concatenate in new band dimension
                    ds = xr.concat([ds, ds_file], dim="band")
                del ds_file # delete to avoid memory problems
                gc.collect() # Call the garbage collector to avoid memory problems

        # Fix band naming
        ds = ds.assign_coords(band=list(GOES_CHANNELS.values()))

        # # open multiple files as a single dataset
        # ds = [xr.open_mfdataset(ifile, preprocess=self.preprocess_fn_radiances, concat_dim="band", combine="nested") for
        #       ifile in files]
        
        # # reinterpolate to match coordinates of the first image
        # ds = [ds[0]] + [ids.interp(x=ds[0].x, y=ds[0].y) for ids in ds[1:]]
        # # concatenate in new band dimension
        # ds = xr.concat(ds, dim="band")

        # Correct latitude longitude assignment after multiprocessing
        # ds['latitude'] = ds.latitude.isel(band=0)
        # ds['longitude'] = ds.longitude.isel(band=0)

        # Keep only certain relevant attributes
        attrs_rad = ds["Rad"].attrs

        ds["Rad"].attrs = {}
        ds["Rad"].attrs = dict(
            long_name=attrs_rad["long_name"],
            standard_name=attrs_rad["standard_name"],
            units=attrs_rad["units"],
        )
        # ds["DQF"].attrs = {}
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

    def preprocess_files(self, skip_if_exists: bool = True):
        """
        Preprocesses multiple files in read path and saves processed files to save path.
        """
        # get unique times from read path
        unique_times = list(set(map(parse_goes16_dates_from_file, self.goes_files)))

        pbar_time = tqdm(unique_times)

        for itime in pbar_time:

            itime_name = format_goes_dates(itime)
            save_filename = Path(self.save_path).joinpath(f"{itime_name}_goes16.nc")
            # skip if file already exists
            if skip_if_exists and os.path.exists(save_filename):
                logger.info(f"File already exists. Skipping: {save_filename}")
                continue

            pbar_time.set_description(f"Processing: {itime}")

            # get files from unique times
            files = list(filter(lambda x: itime in x, self.goes_files))
            try:
                # load radiances
                ds = self.preprocess_radiances(files)
            except:
                logger.error(f"Skipping {itime} due to missing bands")
                continue
            try:
                # load cloud mask
                ds_clouds = self.preprocess_cloud_mask(files)["cloud_mask"]
            except:
                logger.error(f"Skipping {itime} due to missing cloud mask")
                continue
            pbar_time.set_description(f"Loaded data...")
            # interpolate cloud mask to data
            # fill in zeros for all nan values
            ds_clouds = ds_clouds.fillna(0)
            # NOTE: Interpolation changes values from integers to floats
            # NOTE: This is fixed through rounding 
            ds_clouds = ds_clouds.interp(x=ds.x, y=ds.y)
            ds_clouds = ds_clouds.round()

            # save cloud mask as data coordinate
            ds = ds.assign_coords({"cloud_mask": (("y", "x"), ds_clouds.values.squeeze())})
            ds["cloud_mask"].attrs = ds_clouds.attrs

            # check if save path exists, and create if not
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            
            # save to netcdf
            pbar_time.set_description(f"Saving to file...:{save_filename}")
            ds.to_netcdf(save_filename, engine="netcdf4")
            del ds # delete to avoid memory problems
            gc.collect() # Call the garbage collector to avoid memory problems

def geoprocess(
        resolution: float = None, # defined in meters
        read_path: str = "./",
        save_path: str = "./",
        region: str = None,
        resample_method: str = "bilinear",
        skip_if_exists: bool = True
):
    """
    Geoprocesses GOES 16 files

    Args:
        resolution (float, optional): The resolution in meters to resample data to. Defaults to None.
        read_path (str, optional): The path to read the files from. Defaults to "./".
        save_path (str, optional): The path to save the geoprocessed files to. Defaults to "./".
        region (str, optional): The geographic region to extract ("lon_min, lat_min, lon_max, lat_max"). Defaults to None.
        resample_method (str, optional): The resampling method to use. Defaults to "bilinear".
        skip_if_exists (bool, optional): Whether to skip if the file already exists. Defaults to True.
        
    Returns:
        None
    """
    logger.debug(f"Read Path: {read_path}")
    logger.debug(f"Save Path: {save_path}")
    # Initialize GOES 16 GeoProcessor
    logger.info(f"Initializing GOES16 GeoProcessor...")
    # Extracting region from str
    if region is not None:
        region = tuple(map(lambda x: int(x), region.split(" ")))

    goes16_geoprocessor = GOES16GeoProcessing(
        resolution=resolution, 
        read_path=read_path, 
        save_path=save_path,
        region=region,
        resample_method=resample_method
        )
    logger.info(f"GeoProcessing Files...")
    goes16_geoprocessor.preprocess_files(skip_if_exists=skip_if_exists)

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

    # =========================
    # FAILURE TEST CASES
    # =========================
    python geoprocessor_goes16.py --read-path "/home/data" --save-path /home/data/goes/geoprocessed --resolution 5000 --region -200 -15 90 5
    """
    typer.run(geoprocess)
