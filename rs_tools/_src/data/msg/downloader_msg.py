import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import List

from rs_tools import msg_download

import typer
from loguru import logger

@dataclass
class MSGDownload:
    """Downloading class for MSG data and cloud mask"""
    start_date: str
    end_date: str 
    start_time: str 
    end_time: str 
    daily_window_t0: str  
    daily_window_t1: str  
    time_step: str
    save_dir: str 

    def download(self) -> List[str]:
        msg_files = msg_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time,
            end_time=self.end_time,
            daily_window_t0=self.daily_window_t0, 
            daily_window_t1=self.daily_window_t1, 
            time_step=self.time_step,
            satellite="MSG",
            save_dir=Path(self.save_dir).joinpath("L1b"),
            instrument="HRSEVIRI",
            processing_level='L1',
        )
        return msg_files
    
    def download_cloud_mask(self) -> List[str]:
        msg_files = msg_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time,
            end_time=self.end_time,
            daily_window_t0=self.daily_window_t0, 
            daily_window_t1=self.daily_window_t1, 
            time_step=self.time_step,
            satellite="MSG",
            save_dir=Path(self.save_dir).joinpath("CM"),
            instrument="CLM",
            processing_level='L1',
        )
        return msg_files
    

def download(
        start_date: str="2020-10-02",
        end_date: str="2020-10-02", 
        start_time: str="14:00:00", 
        end_time: str="20:00:00", 
        daily_window_t0: str="14:00:00",  
        daily_window_t1: str="14:30:00",  
        time_step: str="00:15:00",
        save_dir: str='./data/', 
        cloud_mask: bool = True
):
    """
    Downloads MSG data including cloud mask

    Args:
        start_date (str): The start date of the data to download (format: 'YYYY-MM-DD')
        end_date (str): The end date of the data to download (format: 'YYYY-MM-DD')
        start_time (str): The start time of the data to download (format: 'HH:MM:SS')
        end_time (str): The end time of the data to download (format: 'HH:MM:SS')
        daily_window_t0 (str): The start time of the daily window (format: 'HH:MM:SS')
        daily_window_t1 (str): The end time of the daily window (format: 'HH:MM:SS')
        time_step (str): The time step between consecutive data downloads (format: 'HH:MM:SS')
        save_dir (str): The path to save the downloaded data
        cloud_mask (bool, optional): Whether to download the cloud mask data (default: True)

    Returns:
        List[str]: List of downloaded file names
    """
    # Initialize MSG Downloader
    logger.info("Initializing MSG Downloader...")
    dc_msg_download = MSGDownload(
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        daily_window_t0=daily_window_t0,
        daily_window_t1=daily_window_t1,
        time_step=time_step,
        save_dir=Path(save_dir).joinpath("msg"),
    )
    logger.info("Downloading MSG Data...")
    msg_filenames = dc_msg_download.download()
    logger.info("Done!")
    if cloud_mask:
        logger.info("Downloading MSG Cloud Mask...")
        msg_filenames = dc_msg_download.download_cloud_mask()
        logger.info("Done!")

    logger.info("Finished MSG Downloading Script...")

if __name__ == '__main__':
    typer.run(download)