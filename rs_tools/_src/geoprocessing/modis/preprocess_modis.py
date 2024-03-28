import autoroot
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
from rs_tools._src.data.modis.reproject import add_modis_crs
import pandas as pd
from datetime import datetime

@dataclass
class GeoProcessingParams:
    # region of interest
    # target grid
    # unit conversion
    # crs transformation
    # resampling Transform
    # region: Tuple[float, float, float, float] = 
    save_path: str = 'path/to/bucket' # analysis ready bucket

def parse_modis_dates_from_file(file: str):
    # get the date from the file
    date = Path(file).name.split(".")[1][1:]
    # get the time from the file
    time = Path(file).name.split(".")[2]
    datetime_str = f"{date}.{time}"
    return datetime_str

@dataclass
class MODISGeoProcessing:
    resolution: float
    read_path: str
    save_path: str

    @property
    def modis_files(self) -> List[MODISFileName]:
        # get a list of all MODIS filenames within the path
        files = get_list_filenames(self.read_path, ".hdf")
        return files
    
    def preprocess_fn(self, ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        # copy to avoid modifying original dataset
        ds = ds.copy() 

        # assign coordinate reference system
        ds = add_modis_crs(ds)

        # TODO: Add functionality to resample data to specific resolution

        return ds

    def preprocess_fn_radiances(self, file: str) -> xr.Dataset:
        # Load file using satpy scenes
        scn = Scene(
            reader="modis_l1b",
            filenames=[file]
        )
        # Load radiance bands
        channels = get_modis_channel_numbers()
        scn.load(channels, generate=False, calibration='radiance')
        
        # change to xarray data
        ds = scn.to_xarray_dataset()  

        # do core preprocess function (e.g. resample, add crs etc.)
        ds = self.preprocess_fn(ds) 

        channels = get_modis_channel_numbers()
        # Store the attributes in a list before concatenation
        attrs_list = [ds[f'CHANNEL_{x}'].attrs for x in channels]

        # concatinate in new band dimension
        # TODO: Concatination overwrites the attributes of original bands
        ds = xr.concat(list(map(lambda x: ds[f'CHANNEL_{x}'], channels)), dim="band")
        # rename band dimensions
        ds = ds.assign_coords(band=list(map(lambda x: x, channels)))

        # Reassign the attributes to each band
        for band, attrs in zip(ds['band'].values, attrs_list):
            ds.sel(band=band).attrs = attrs

        # convert measurement time (in seconds) to datetime
        time_stamp = pd.to_datetime(ds.start_time)
        time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")  
        # assign bands and time data to each variable
        ds = ds.expand_dims({"time": [time_stamp]})
        
        return ds

    def preprocess_fn_cloudmask(self, ds: xr.Dataset) -> xr.Dataset:
        pass

    def preprocess_radiances(self, files: List[str]) -> xr.Dataset:
        identifier = MODIS_NAME_TO_ID[self.satellite]
        
        # Check that all files contain radiance data
        file = list(filter(lambda x: identifier in x, files))

        # Check that only one file is selected
        logger.info(f"Number of radiance files: {len(files)}")
        assert len(file) == 1

        # load file using satpy, convert to xarray dataset, and preprocess
        ds = self.preprocess_fn_radiances(file)

        # assign band wavelength to each variable
        ds = ds['band'].expand_dims({"band_wavelength": ds.band_wavelength.values})
        ds_subset.band_wavelength.attrs = ds.band_wavelength.attrs

        # NOTE: Keep only certain relevant attributes
        attrs_rad = ds.attrs

        ds.attrs = {}
        ds.attrs = dict(
            calibration=attrs_rad["calibration"],
            long_name=attrs_rad["platform_name"],
            reader=attrs_rad["reader"],
            standard_name=attrs_rad["standard_name"],
            units=attrs_rad["units"],
            wavelength=attrs_rad["wavelength"]
        )
        
        return ds


    def preprocess_cloud_mask(self, files: List[str]) -> xr.Dataset:
        identifier = MODIS_NAME_TO_ID[f'{self.satellite}_cloud']
        
        # Check that all files contain radiance data
        file = list(filter(lambda x: identifier in x, files))

        # Check that only one file is selected
        logger.info(f"Number of cloud mask files: {len(files)}")
        assert len(file) == 1

        # Load file using satpy scenes
        scn = Scene(
            reader="modis_l2",
            filenames=[file]
        )
        # Load cloud mask data
        datasets = scn.available_dataset_names()
        # Needs to be loaded at 1000 m resolution for all channels to match
        scn.load(datasets, generate=False, resolution=1000) 
        
        # change to xarray data
        ds = scn.to_xarray_dataset()

        return ds


    def preprocess_files(self):
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
                logger.error(f"Skipping {itime} due to missing bands")
                continue
            try:
                # load cloud mask
                ds_clouds = self.preprocess_cloud_mask(files)["cloud_mask"]
            except AssertionError:
                logger.error(f"Skipping {itime} due to missing cloud mask")
                continue

            # # interpolate cloud mask to data
            # ds_clouds = ds_clouds.interp(x=ds.x, y=ds.y)
            # # save cloud mask as data coordinate
            # ds = ds.assign_coords({"cloud_mask": (("y", "x"), ds_clouds.values.squeeze())})
            # ds["cloud_mask"].attrs = ds_clouds.attrs

            # # check if save path exists, and create if not
            # if not os.path.exists(self.save_path):
            #     os.makedirs(self.save_path)
        
            # # remove file if it already exists
            # save_filename = Path(self.save_path).joinpath(f"{itime}_goes16.nc")
            # if os.path.exists(save_filename):
            #     logger.info(f"File already exists. Overwriting file: {save_filename}")
            #     os.remove(save_filename)
            # # save to netcdf
            # ds.to_netcdf(save_filename, engine="netcdf4")

    def load_modis_files(self, satellite: str="aqua"):

        # load MODIS SCENE
        # get paired files for satellites
        paired_files = get_modis_paired_files(self.modis_files, satellite)

        pbar = tqdm(list(paired_files.keys()))

        for itime in pbar:
            pbar.set_description(f"Time: {itime}")
            

            ds_modis = ds_modis.assign_coords({"cloud_mask": (("y", "x"), scn["cloud_mask"].values)})
            # add time dimensions
            time_stamp = datetime.strptime(itime, "%Y%m%d%H%M")
            ds_modis = ds_modis.expand_dims(time=[time_stamp])
            ds_modis = ds_modis.drop_vars("crs")
            # TODO: keep important attributes
            useful_keys = ["calibration", "wavelength", "standard_name"]
            ds_modis.attrs = {k: v for k, v in ds_modis.attrs.items() if k in useful_keys}
            # rename
            ds_modis.name = ds_modis.attrs["calibration"]
            ds_modis.to_netcdf(Path(self.save_path).joinpath(f"{itime}_{satellite}.nc"), engine="netcdf4")


def geoprocess_modis(
        resolution: float = 1000,
        read_path: str = "/Users/anna.jungbluth/Desktop/git/rs_tools/data/aqua",
        save_path: str = "/Users/anna.jungbluth/Desktop/git/rs_tools/data/aqua/geoprocessed"
):
    """
    Downloads MODIS TERRA and GOES 16 files for the specified period, region, and save path.

    Args:
        period (List[str], optional): The period of time to download files for. Defaults to ["2020-10-01", "2020-10-31"].
        region (Tuple[str], optional): The geographic region to download files for. Defaults to (-180, -90, 180, 90).
        save_path (str, optional): The path to save the downloaded files. Defaults to "./".

    Returns:
        None
    """
    logger.info(f"Starting Script...")

    logger.info(f"Initializing GeoProcessor...")
    modis_geoprocessor = MODISGeoProcessing(
        resolution=resolution, read_path=read_path, save_path=save_path
        )
    logger.info(f"Saving Files...")
    modis_geoprocessor.save_files()
    
    # out_files = []
    # pbar = tqdm(modis_geoprocessor.unique_files)
    # for ifile in pbar:
    #     pbar.set_description(f"Processing File: {ifile}")
    #     out_files.append(modis_geoprocessor.geoprocess_file(ifile))

    logger.info(f"Finished Script...!")




if __name__ == '__main__':
    """
    python scripts/pipeline/preprocess_modis.py --read-path "/home/juanjohn/data/rs/modis/raw" --save-path /home/juanjohn/data/rs/modis/analysis
    """
    typer.run(geoprocess_modis)
