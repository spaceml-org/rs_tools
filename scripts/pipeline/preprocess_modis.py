import autoroot
import numpy as np

import rioxarray
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

from rs_tools import goes_download, modis_download, MODIS_VARIABLES, MODIS_CHANNEL_NUMBERS
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.grid import create_latlon_grid
import typer
from loguru import logger
import xarray as xr
from satpy import Scene


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
    # resolution: float = 0.25 # in degrees?
    resolution: float = 1_000 # km
    save_path: str = "./"
    channels: list[str] = MODIS_CHANNEL_NUMBERS

    # def geoprocess_files(self, params: GeoProcessingParams, files: List[str]):
    # resolution: float = 0.25 # in degrees?
    # save_path: str = "./"

    # def geoprocess(self, params: GeoProcessingParams, files: List[str]):
    #     save_path: str = params.save_path.join(self.save_path)
    #     target_grid: np.ndarray = create_latlon_grid(params.region, self.resolution) # create using np.linspace?

    #     # loop through files
    #     # open dataset
    #     # stack variables to channels
    #     # resample
    #     # convert units (before or after resampling???)
    #     # save as netcdf

    def geoprocess_file(self, file: str, save_name: str="test") -> None:

        # load MODIS SCENE
        ds, time_stamp = self.load_modis_scene(file)
        # convert units
        ds = self.convert_units(file)
        # TODO: resample
        # save to raster
        time_stamp = time_stamp.strftime("%Y%m%d%H%M")
        save_name =f"modis_{time_stamp}.tif"
        ds.rio.to_raster(Path(self.save_path).joinpath(save_name))

        return None

    def load_modis_scene(self, file: str) -> xr.Dataset:

        # load modis scene
        scn = Scene(reader="modis_l1b", filenames=[file])

        # load channels
        scn.load(self.channels, resolution = self.resolution)

        time_stamp = scn.start_time + (scn.end_time - scn.start_time) / 2

        # convert to xarray
        ds = scn.to_xarray_dataset()

        return ds, time_stamp

    def convert_units(self, ds: xr.Dataset) -> xr.Dataset:

        ds.rio.write_crs("epsg:4326", inplace=True)
        ds = ds.set_coords(["latitude", "longitude"])
        ds = ds.assign_coords({"latitude": ds.latitude, "longitude": ds.longitude})
        return ds






def preprocess_modis(
        modis_save_dir: str = "./",
        start_date: str = "2020-10-01",
        end_date: str = "2020-10-31",
        resolution: float = 0.25,
        region: str = "-180 -90 180 90",
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
    
    # load files from directory
    list_of_files = get_list_filenames(modis_save_dir, "hdf")
    logger.info(f"{len(list_of_files)} to process...")

    # open file
    ifile = list_of_files[-1]
    print(ifile)
    out = xr.open_dataset(ifile, engine="netcdf4")
    pass




if __name__ == '__main__':
    """
    python scripts/pipeline/preprocess_modis.py --modis-save-dir "/home/juanjohn/projects/rs_tools/modis/"
    """
    typer.run(preprocess_modis)
