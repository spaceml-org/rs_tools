import autoroot
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

from rs_tools import goes_download, modis_download, MODIS_VARIABLES
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.grid import create_latlon_grid
import typer
from loguru import logger
import xarray as xr


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
    resolution: float = 0.25 # in degrees?
    save_path: str = "./"

    def geoprocess(self, params: GeoProcessingParams, files: List[str]):
        save_path: str = params.save_path.join(self.save_path)
        target_grid: np.ndarray = create_latlon_grid(params.region, self.resolution) # create using np.linspace?

        # loop through files
        # open dataset
        # stack variables to channels
        # resample
        # convert units (before or after resampling???)
        # save as netcdf







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
