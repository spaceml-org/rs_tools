import typer
import tqdm
import autoroot
from loguru import logger
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import List

from rs_tools import msg_download
from rs_tools._src.data.modis import _check_earthdata_login, modis_granule_to_datetime, query_modis_timestamps
from rs_tools._src.data.msg import MSGFileName

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
        modis_product: str="MYD021KM", # TODO: Add advanced data product mapping
        start_date: str="2020-10-02",
        end_date: str="2020-10-02", 
        start_time: str="00:00:00", 
        end_time: str="00:15:00", 
        save_dir: str='./data/', 
        cloud_mask: bool = True
):
    """
    Downloads MSG data including cloud mask 

    Args:
        modis_product (str): The MODIS product to download (default: 'MYD021KM', i.e. Aqua at 1km resolution)
        start_date (str): The start date of the data to download (format: 'YYYY-MM-DD')
        end_date (str): The end date of the data to download (format: 'YYYY-MM-DD')
        start_time (str): The start time of the data to download (format: 'HH:MM:SS')
        end_time (str): The end time of the data to download (format: 'HH:MM:SS')
        save_dir (str): The path to save the downloaded data
        cloud_mask (bool, optional): Whether to download the cloud mask data (default: True)

    Returns:
        List[str]: List of downloaded file names
    """
    list_modis_times = []
    list_msg_times = []
    list_msg_cm_times = []

    logger.info("Querying MODIS overpasses for MSG field-of-view and specified time period...")
    # Check EartData login
    _check_earthdata_login()
    # Compile MODIS timestamp tuple
    start_datetime_str = start_date + ' ' + start_time
    end_datetime_str = end_date + ' ' + end_time
    # Query MODIS timestamps
    modis_results = query_modis_timestamps(
        short_name=modis_product,
        bounding_box=(-70, -70, 70, 70), # Approximate field of view of MSG
        temporal=(start_datetime_str, end_datetime_str)
    )
    logger.info(f"Found {len(modis_results)} MODIS granules for MSG field-of-view and specified time period...")
    # Extract MODIS timestamps
    modis_timestamps = [modis_granule_to_datetime(x) for x in modis_results]

    # create progress bar for dates
    pbar_time = tqdm.tqdm(modis_timestamps)

    # Initialize MSG Downloader
    logger.info("Initializing MSG Downloader...")

    for itime in pbar_time:
        pbar_time.set_description(f"Time - {itime}")
        end_itime = itime + timedelta(minutes=10)


        dc_msg_download = MSGDownload(
            start_date=str(itime.date()),
            end_date=str(end_itime.date()),
            start_time=str(itime.time()),
            end_time=str(end_itime.time()),
            daily_window_t0="00:00:00",
            daily_window_t1="23:59:00",
            time_step="00:15:00",
            save_dir=Path(save_dir).joinpath("msg"),
        )

        msg_filenames = dc_msg_download.download()

        if cloud_mask:
            msg_cm_filenames = dc_msg_download.download_cloud_mask()

        if len(msg_filenames) > 0: # Check if any files were downloaded
            assert len(msg_filenames) == 1, "More than one MSG file was downloaded"
            list_modis_times.append(str(itime))
            
            msg_filename = str(MSGFileName.from_filename(msg_filenames[0]).datetime_acquisition)
            list_msg_times.append(msg_filename)

            if cloud_mask:
                assert len(msg_filenames) == len(msg_cm_filenames), "Different number of MSG and cloud mask files downloaded"
                msg_cm_filename = str(MSGFileName.from_filename(msg_cm_filenames[0]).datetime_acquisition)
                list_msg_cm_times.append(msg_cm_filename)
    
    logger.info("Finished Data Download...")
    # Compile summary file
    logger.info("Compiling summary file...")
    df = pd.DataFrame()
    df['MODIS'] = list_modis_times
    df['MSG'] = list_msg_times
    if cloud_mask:
        df['MSG_cloudmask'] = list_msg_cm_times
    df.to_csv(Path(save_dir).joinpath("msg-modis-timestamps.csv"), index=False)

    logger.info("Done!")
    logger.info("Finished MSG Downloading Script...")

if __name__ == '__main__':
    typer.run(download)