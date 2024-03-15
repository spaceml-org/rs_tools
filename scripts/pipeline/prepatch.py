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



@dataclass(frozen=True)
class PrePatcher:
    """
    A class for preprocessing and saving patches from NetCDF files.

    Attributes:
        read_path (str): The path to the directory containing the NetCDF files.
        patch_size (int): The size of each patch.
        stride_size (int): The stride size for generating patches.
        save_path (str): The path to save the patches.

    Methods:
        nc_files(self) -> List[str]: Returns a list of all NetCDF filenames in the read_path directory.
        save_patches(self): Preprocesses and saves patches from the NetCDF files.
    """

    read_path: str = "./"
    patch_size: int = 256
    stride_size: int = 256
    save_path: str = "./"

    @property
    def nc_files(self) -> List[str]:
        """
        Returns a list of all NetCDF filenames in the read_path directory.

        Returns:
            List[str]: A list of NetCDF filenames.
        """
        # get a list of all filenames within the path
        # get all modis files
        modis_files = get_list_filenames(self.read_path, ".nc")
        return modis_files

    def save_patches(self):
        """
        Preprocesses and saves patches from the NetCDF files.
        """
        pbar = tqdm(self.nc_files)

        save_path = str(Path(self.save_path))

        for ifile in pbar:
            # open dataset
            itime = str(Path(ifile).name).split("_")[0]
            ds = xr.open_dataarray(ifile, engine="netcdf4")
            patches = dict(x=self.patch_size, y=self.patch_size)
            strides = dict(x=self.stride_size, y=self.stride_size)
            patcher = XRDAPatcher(da=ds, patches=patches, strides=strides)
            for i, ipatch in tqdm(enumerate(patcher), total=len(patcher)):
                # save as numpy files
                np.save(Path(save_path).joinpath(f"{itime}_radiance_patch_{i}"), ipatch.values)
                np.save(Path(save_path).joinpath(f"{itime}_latitude_patch_{i}"), ipatch.latitude.values)
                np.save(Path(save_path).joinpath(f"{itime}_longitude_patch_{i}"), ipatch.longitude.values)
                np.save(Path(save_path).joinpath(f"{itime}_cloudmask_patch_{i}"), ipatch.cloud_mask.values)

def prepatch(
        read_path: str = "./",
        patch_size: int = 256,
        stride_size: int = 256,
        save_path: str = "./"
):
    """
    Patches satellite data into smaller patches for training.
    Args:
        read_path (str, optional): The path to read the input files from. Defaults to "./".
        patch_size (int, optional): The size of each patch. Defaults to 256.
        stride_size (int, optional): The stride size for patch extraction. Defaults to 256.
        save_path (str, optional): The path to save the extracted patches. Defaults to "./".

    Returns:
        None
    """
    logger.info(f"Starting Script...")

    logger.info(f"Initializing Prepatcher...")
    prepatcher = PrePatcher(
        read_path=read_path, save_path=save_path
        )
    logger.info(f"Saving Files...: {save_path}")
    prepatcher.save_patches()
    
    # out_files = []
    # pbar = tqdm(modis_geoprocessor.unique_files)
    # for ifile in pbar:
    #     pbar.set_description(f"Processing File: {ifile}")
    #     out_files.append(modis_geoprocessor.geoprocess_file(ifile))

    logger.info(f"Finished Script...!")


if __name__ == '__main__':
    """
    python scripts/pipeline/prepatch.py --read-path "/path/to/netcdf/file" --save-path /path/to/save/patches
    """
    typer.run(prepatch)
