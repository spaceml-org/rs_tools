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
    """Downloading class for GOES 16 data and cloud mask"""
    start_date: str
    end_date: str 
    start_time: str 
    end_time: str 
    daily_window_t0: str  
    daily_window_t1: str  
    time_step: str
    save_dir: str 

    def download(self) -> List[str]:
        goes_files = goes_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time,
            end_time=self.end_time,
            daily_window_t0=self.daily_window_t0, 
            daily_window_t1=self.daily_window_t1, 
            time_step=self.time_step,
            satellite_number=16,
            save_dir=Path(self.save_dir).joinpath("L1b"),
            instrument="ABI",
            processing_level='L1b',
            data_product='Rad',
            domain='F',
            bands='all',
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
            satellite_number=16,
            save_dir=Path(self.save_dir).joinpath("CM"),
            instrument="ABI",
            processing_level='L2',
            data_product='ACM',
            domain='F',
            bands='all',
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
        save_dir: str, 
        cloud_mask: bool = True
):
    """
    Downloads GOES 16 data including cloud mask

    Args:
        start_date (str): The start date of the data to download (format: 'YYYY-MM-DD')
        end_date (str): The end date of the data to download (format: 'YYYY-MM-DD')
        start_time (str): The start time of the data to download (format: 'HH:MM:SS')
        end_time (str): The end time of the data to download (format: 'HH:MM:SS')
        daily_window_t0 (str): The start time of the daily window (format: 'HH:MM:SS')
        daily_window_t1 (str): The end time of the daily window (format: 'HH:MM:SS')
        time_step (str): The time step between consecutive data downloads (format: 'HH:MM:SS')
        save_path (str): The path to save the downloaded data
        cloud_mask (bool, optional): Whether to download the cloud mask data (default: True)

    Returns:
        List[str]: List of downloaded file names
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
        save_dir=Path(save_dir).joinpath("goes16"),
    )
    logger.info("Downloading GOES 16 Data...")
    goes16_filenames = dc_goes16_download.download()
    logger.info("Done!")
    if cloud_mask:
        logger.info("Downloading GOES 16 Cloud Mask...")
        goes16_filenames = dc_goes16_download.download_cloud_mask()
        logger.info("Done!")

    logger.info("Finished GOES 16 Downloading Script...")

if __name__ == '__main__':
    typer.run(download)