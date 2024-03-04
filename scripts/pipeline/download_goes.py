"""
A General Pipeline for create ML-Ready Data
- Downloading the Data
- Data Harmonization
- Normalizing
- Patching
"""
import autoroot
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

from rs_tools import goes_download, modis_download, MODIS_VARIABLES

from rs_tools._src.geoprocessing.grid import create_latlon_grid
import typer
from loguru import logger


@dataclass
class DownloadParameters:
    start_date: str = "2020-10-01"
    end_date: str = "2020-10-02"
    region: Tuple[float, float, float, float] = (-180, -90, 180, 90)
    save_path: str = "./"



@dataclass
class GOES16Download:
    """GOES will save to separate subdirectories"""
    channels: str = "all"
    satellite: int = 16
    start_date: str = "2020-10-01"
    end_date: str = "2020-10-31"
    start_time: str = '00:00:00'
    end_time: str = '23:59:00'
    daily_window_t0: str = '14:00:00' # Times in UTC, 9 AM local time
    daily_window_t1: str = '20:00:00' # Times in UTC, 3 PM local time
    time_step: str = "12:00:00"
    save_path: str = "./goes"

    def download(self) -> List[str]:
        goes_files = goes_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time,
            end_time=self.end_time,
            daily_window_t0=self.daily_window_t0, # Times in UTC, 9 AM local time
            daily_window_t1=self.daily_window_t1, # Times in UTC, 3 PM local time
            time_step=self.time_step,
            satellite_number=self.satellite,
            save_dir=self.save_path,
            instrument="ABI",
            processing_level='L1b',
            data_product='Rad',
            domain='F',
            bands=self.channels,
            check_bands_downloaded=True,
        )
        return goes_files
    
    def download_cloud_mask(self, params: DownloadParameters) -> List[str]:
        return None
    

def download(
        start_date: str = "2020-10-01",
        end_date: str = "2020-10-31",
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
    region = tuple(map(lambda x: int(x), region.split(" ")))

    # initialize GOES 16 Files
    logger.info("Initializing GOES16 parameters...")
    dc_goes16_download = GOES16Download(
        start_date=params.start_date,
        end_date=params.end_date,
        save_path=str(Path(params.save_path).joinpath("goes16"))
    )
    logger.info("Downloading GOES 16...")
    goes16_filenames = dc_goes16_download.download()
    logger.info("Done!")

    # TODO: Create DataFrame
    # TODO: save MODIS-TERRA filenames to CSV?
    # TODO: save GOES-16 filenames to CSV?
    logger.info("Finished Script...")


if __name__ == '__main__':
    typer.run(download)