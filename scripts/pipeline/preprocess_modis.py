import autoroot
import numpy as np
from xrpatcher._src.base import XRDAPatcher
import rioxarray
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from tqdm import tqdm
from rs_tools import goes_download, modis_download, MODIS_VARIABLES, get_modis_channel_numbers
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.grid import create_latlon_grid
import typer
from loguru import logger
import xarray as xr
from satpy import Scene
import datetime
from rs_tools._src.data.modis import MODISFileName, MODIS_ID_TO_NAME, MODIS_NAME_TO_ID, get_modis_paired_files
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



@dataclass
class MODISGeoProcessing:
    resolution: float = 1_000 # km
    read_path: str = "./"
    patch_size: int = 256
    stride_size: int = 256
    save_path: str = "./"

    @property
    def modis_files(self) -> List[MODISFileName]:
        # get a list of all filenames within the path
        # get all modis files
        modis_files = get_list_filenames(self.read_path, ".hdf")
        modis_filenames = list(map(lambda x: MODISFileName.from_filename(x), modis_files))
        return modis_filenames

    def save_files(self):

        # load MODIS SCENE
        # get paired files for satellites
        # self.load_modis_files("aqua")
        self.load_modis_files("terra")

    def load_modis_files(self, satellite: str="aqua"):

        # load MODIS SCENE
        # get paired files for satellites
        paired_files = get_modis_paired_files(self.modis_files, satellite)

        pbar = tqdm(list(paired_files.keys()))

        for itime in pbar:
            pbar.set_description(f"Time: {itime}")
            scn = Scene(
                reader="modis_l1b",
                filenames=[
                    str(paired_files[itime]["data"].full_path),
                ], 
                reader_kwargs={"calibration": "radiances"}
            )
            scn.load(get_modis_channel_numbers(), resolution = self.resolution)
            
            # change to xarray data
            ds_modis = scn.to_xarray_dataset()
            # create new channel dimension
            ds_modis = xr.concat(list(map(lambda x: ds_modis[x], get_modis_channel_numbers())), dim="channel")
            # rename channel dimensions
            ds_modis = ds_modis.assign_coords(channel=list(map(lambda x: x, get_modis_channel_numbers())))

            # Add Cloud Mask
            scn = Scene(
                reader="modis_l2", 
                filenames=[
                    str(paired_files[itime]["cloud"].full_path),
                ], 
                reader_kwargs={"calibration": "radiance"}
            )
            scn.load(["cloud_mask"], resolution = self.resolution)
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


    # def save_patches(self, ds):
    #     patches = dict(x=self.patch_size, y=self.patch_size)
    #     strides = dict(x=self.stride_size, y=self.stride_size)
    #     patcher = XRDAPatcher(da=ds, patches=patches, strides=strides)
    #     for i, ipatch in tqdm(enumerate(patcher), total=len(patcher)):
    #         # save as numpy files
    #         np.savez(
    #             Path(self.save_path).joinpath(f"{ipatch.time.values}_patch{i}.npz"),
    #             data=ipatch.values, 
    #             lat=ipatch.latitude.values,
    #             lon=ipatch.longitude.values,
    #             cloud_mask=ipatch.latitude.values
    #         )

    # def load_modis_scene(self, file: str) -> Tuple[xr.Dataset, datetime.datetime]:

    #     # load modis scene
    #     scn = Scene(reader="modis_l1b", filenames=[file])

    #     # load channels
    #     channels = get_modis_channel_numbers()
    #     scn.load(channels, resolution = self.resolution)

    #     time_stamp = scn.start_time + (scn.end_time - scn.start_time) / 2

    #     # convert to xarray
    #     ds = scn.to_xarray_dataset()

    #     ds = ds.assign_coord({"time": time_stamp})

    #     return ds, time_stamp

    # def convert_units(self, ds: xr.Dataset) -> xr.Dataset:

    #     ds.rio.write_crs("epsg:4326", inplace=True)
    #     ds = ds.set_coords(["latitude", "longitude"])
    #     ds = ds.assign_coords({"latitude": ds.latitude, "longitude": ds.longitude})
    #     return ds



def preprocess_modis(
        modis_save_dir: str = "./",
        resolution: float = 1000,
        read_path: str = "./",
        patch_size: int = 256,
        stride_size: int = 256,
        save_path: str = "./"
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
    typer.run(preprocess_modis)
