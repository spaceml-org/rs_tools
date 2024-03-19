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
from typing import List
from rs_tools import modis_download

import typer
from loguru import logger


@dataclass
class MODISAquaDownload:
    """Downloading class for AQUA/MODIS data and cloud mask"""
    start_date: str
    end_date: str
    start_time: str
    end_time: str
    save_dir: str
    bounding_box: tuple[float, float, float, float]
    
    def download(self) -> List[str]:
        aqua_files = modis_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time, 
            end_time=self.end_time,
            day_step=1,
            satellite="Aqua",
            save_dir=Path(self.save_dir).joinpath("L1b"),
            processing_level='L1b',
            resolution="1KM",
            bounding_box=self.bounding_box,
            day_night_flag="day",
            identifier= "02"
            )
        return aqua_files
    
    def download_cloud_mask(self) -> List[str]:
        aqua_files = modis_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time, 
            end_time=self.end_time,
            day_step=1,
            satellite="Aqua",
            save_dir=Path(self.save_dir).joinpath("CM"),
            processing_level='L2',
            resolution="1KM",
            bounding_box=self.bounding_box,
            day_night_flag="day",
            identifier= "35"
            )
        return aqua_files


def download(
        start_date: str = "2018-10-01", 
        end_date: str = "2018-10-01",
        start_time: str = "14:00:00",
        end_time: str = "21:00:00",
        save_dir: str = "./data/",
        region: str = "-130 -15 -90 5",
        cloud_mask: bool = True
):
    """
    Downloads AQUA MODIS data including cloud mask

    Args:
        start_date (str): The start date of the period to download files for in the format "YYYY-MM-DD".
        end_date (str): The end date of the period to download files for in the format "YYYY-MM-DD".
        start_time (str): The start time of the period to download files for in the format "HH:MM:SS".
        end_time (str): The end time of the period to download files for in the format "HH:MM:SS".
        save_dir (str): The directory path to save the downloaded files.
        region (str, optional): The geographic region to download files for in the format "min_lon min_lat max_lon max_lat".
        cloud_mask (bool, optional): Whether to download the cloud mask data (default: True).

    Returns:
        None
    """
    bounding_box = tuple(map(lambda x: int(x), region.split(" ")))
    # Initialize AQUA Downloader
    logger.info("Initializing AQUA Downloader...")
    dc_aqua_download = MODISAquaDownload(
        start_date=start_date,
        end_date=end_date,
        start_time=start_time,
        end_time=end_time,
        save_dir=Path(save_dir).joinpath("aqua"),
        bounding_box=bounding_box,
    )
    logger.info("Downloading AQUA Data...")
    modis_filenames = dc_aqua_download.download()
    logger.info("Done!")
    if cloud_mask:
        logger.info("Downloading AQUA Cloud Mask...")
        modis_filenames = dc_aqua_download.download_cloud_mask()
        logger.info("Done!")

    logger.info("Finished AQUA Downloading Script...")

if __name__ == '__main__':
    typer.run(download)