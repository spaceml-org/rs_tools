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
class MODISTerraDownload:
    """MODIS will save to 1 subdirectory"""
    start_date: str = "2020-10-01"
    end_date: str = "2020-10-31"
    start_time: str = '14:00:00' # Times in UTC, 9 AM local time
    end_time: str = '20:00:00' # Times in UTC, 3 PM local time
    region: Tuple[float, float, float, float] = (-180, -90, 180, 90)
    save_path: str = "./modis"
    
    def download(self) -> List[str]:
        modis_files = modis_download(
            start_date=self.start_date,
            end_date=self.end_date,
            start_time=self.start_time, # used for daily window
            end_time=self.end_time, # used for daily window
            day_step=1,
            satellite="Terra",
            save_dir=self.save_path,
            processing_level='L1b',
            resolution="1KM",
            bounding_box=self.region,
            day_night_flag="day",
            identifier= "02"
        )
        return modis_files

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
    # initialize params
    logger.info("Initializing MODIS parameters...")
    params = DownloadParameters(start_date=start_date, end_date=end_date, region=region, save_path=save_path)
    # initialize MODIS TERRA Files downloader
    dc_modis_download = MODISTerraDownload(
        start_date=params.start_date,
        end_date=params.end_date,
        region=params.region,
        save_path=str(Path(params.save_path).joinpath("modis"))
    )
    logger.info("Downloading MODIS...")
    modis_filenames = dc_modis_download.download()
    logger.info("Done!")

    logger.info("Finished Script...")


if __name__ == '__main__':
    typer.run(download)