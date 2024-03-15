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
from tqdm.auto import tqdm
from rs_tools import goes_download, modis_download, MODIS_VARIABLES

from rs_tools._src.geoprocessing.grid import create_latlon_grid
import typer
from loguru import logger
import earthaccess
import pandas as pd


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
    region: Tuple[float, float, float, float] = (-130, -15, -90, 5)
    save_path: str = "./modis"
    
    def download(self) -> List[str]:
        # modis_files = modis_download(
        #     start_date=self.start_date,
        #     end_date=self.end_date,
        #     start_time=self.start_time, # used for daily window
        #     end_time=self.end_time, # used for daily window
        #     day_step=1,
        #     satellite="Terra",
        #     save_dir=Path(self.save_path).joinpath("L1b"),
        #     processing_level='L1b',
        #     resolution="1KM",
        #     bounding_box=self.region,
        #     day_night_flag="day",
        #     identifier= "02"
        # )

        # TODO: Use our download function instead of earthaccess
        pbar = tqdm(pd.date_range(start=self.start_date, end=self.end_date, freq="D"))
        for idate in pbar:
            pbar.set_description(f"Downloading Terra - Date: {idate}")
            
            # temporal window
            window = (f"{idate} {self.start_time}", f"{idate} {self.end_time}")
            
            results = earthaccess.search_data(
                short_name="MOD021KM",
                cloud_hosted=True,
                bounding_box=self.region,
                temporal=window,
                count=-1
            )
            # download if something found
            if len(results) != 0:
                earthaccess.download(results, self.save_path)

            pbar.set_description(f"Downloading Terra Cloud Mask - Date: {idate}")
            results = earthaccess.search_data(
                short_name="MOD35_L2",
                cloud_hosted=True,
                bounding_box=self.region,
                temporal=window,
                count=-1
            )
            # download if something found
            if len(results) != 0:
                earthaccess.download(results, self.save_path)

    def download_cloud_mask(self, params: DownloadParameters) -> List[str]:
        return None


def download(
        start_date: str = "2020-10-01",
        end_date: str = "2020-10-05",
        region: str = "-130 -15 -90 5",
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