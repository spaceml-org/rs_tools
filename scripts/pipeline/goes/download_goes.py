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
class GOES16Download:
    """GOES will save to separate subdirectories"""
    channels: str = "all"
    satellite: int = 16
    start_date: str = "2018-10-01"
    end_date: str = "2018-10-31"
    start_time: str = '14:00:00'
    end_time: str = '20:00:00'
    daily_window_t0: str = '14:00:00' # Times in UTC, 9 AM local time
    daily_window_t1: str = '20:00:00' # Times in UTC, 3 PM local time
    time_step: str = "4:00:00" # download one image at 14:00 and one at 18:00 every day
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
    
    def download_cloud_mask(self) -> List[str]:
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
            processing_level='L2',
            data_product='ACM',
            domain='F',
            bands=self.channels,
            check_bands_downloaded=True,
        )
        return goes_files
    

def download(
        start_date: str = "2020-10-01",
        end_date: str = "2020-10-02",
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

    params = DownloadParameters(
        start_date=start_date,
        end_date=end_date,
        region=region,
        save_path=save_path
    )

    # initialize GOES 16 Files
    logger.info("Initializing GOES16 parameters...")
    dc_goes16_download = GOES16Download(
        start_date=start_date,
        end_date=end_date,
        save_path=str(Path(save_path).joinpath("goes16"))
    )
    logger.info("Downloading GOES 16...")
    goes16_filenames = dc_goes16_download.download()
    logger.info("Done!")
    logger.info("Downloading GOES 16 Cloud Mask...")
    goes16_filenames = dc_goes16_download.download_cloud_mask()
    logger.info("Done!")

    # TODO: Create DataFrame
    # TODO: save GOES-16 filenames to CSV?
    logger.info("Finished Script...")


if __name__ == '__main__':
    typer.run(download)