import autoroot
import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import List

from rs_tools import goes_download

import typer
from loguru import logger

@dataclass
class GOES16Download:
    """Downloads all available bands for GOES 16"""
    start_date: str
    end_date: str 
    start_time: str 
    end_time: str 
    daily_window_t0: str  
    daily_window_t1: str  
    time_step: str
    save_path: str 
    channels: str = "all"
    satellite: int = 16

    def download(self) -> List[str]:
        goes_files = goes_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time,
            end_time=self.end_time,
            daily_window_t0=self.daily_window_t0, 
            daily_window_t1=self.daily_window_t1, 
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
            daily_window_t0=self.daily_window_t0, 
            daily_window_t1=self.daily_window_t1, 
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
        start_date: str,
        end_date: str, 
        start_time: str, 
        end_time: str, 
        daily_window_t0: str,  
        daily_window_t1: str,  
        time_step: str,
        save_path: str, 
        cloud_mask: bool = True
):
    """
    Downloads GOES 16 Data including cloud mask

    :param start_date: The start date of the data to download (format: 'YYYY-MM-DD')
    :param end_date: The end date of the data to download (format: 'YYYY-MM-DD')
    :param start_time: The start time of the data to download (format: 'HH:MM:SS')
    :param end_time: The end time of the data to download (format: 'HH:MM:SS')
    :param daily_window_t0: The start time of the daily window (format: 'HH:MM:SS')
    :param daily_window_t1: The end time of the daily window (format: 'HH:MM:SS')
    :param time_step: The time step between consecutive data downloads (format: 'HH:MM:SS')
    :param save_path: The path to save the downloaded data
    :param cloud_mask: Whether to download the cloud mask data (default: True)
    :return: None
    """
    # Initialize GOES 16 Downloader
    logger.info("Initializing GOES16 Downloader...")
    dc_goes16_download = GOES16Download(
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        daily_window_t0=daily_window_t0,
        daily_window_t1=daily_window_t1,
        time_step=time_step,
        save_path=str(Path(save_path).joinpath("goes16"))
    )
    logger.info("Downloading GOES 16...")
    goes16_filenames = dc_goes16_download.download()
    logger.info("Done!")
    if cloud_mask:
        logger.info("Downloading GOES 16 Cloud Mask...")
        goes16_filenames = dc_goes16_download.download_cloud_mask()
        logger.info("Done!")

    logger.info("Finished GOES 16 Downloading Script...")

if __name__ == '__main__':
    typer.run(download)