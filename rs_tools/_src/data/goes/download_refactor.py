import autoroot
from typing import Optional, Tuple, List, Union
from tqdm.auto import tqdm
import typer
from pathlib import Path
import pandas as pd
import geopandas as gpd
from odc.geo.geom import BoundingBox
from loguru import logger
import earthaccess
from datetime import datetime
from odc.geo.geom import Geometry, polygon
import geopandas as gpd
import pandas as pd
from rs_tools._src.geoprocessing.geometry import bbox_string_to_bbox
import goes2go
from goes2go import GOES
from rs_tools._src.data.goes.bands import GOES16_BANDS
from dataclasses import dataclass, field
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from rs_tools._src.preprocessing.periods import TimeQuery, TimeSeries

app = typer.Typer()


def goes2go_download_from_timeseries(
    time_series: List[str],
    within: str = "10 minutes",
    product: str = "ABI-L1b-Rad",
    satellite: int = 16,
    domain: str = "F",
    save_dir: str = "./data",
    save_file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Downloads GOES satellite data for the specified time series.

    Args:
        time_series (List[str]): List of time stamps in string format.
        within (str, optional): Time window within which to search for data. Defaults to "10 minutes".
        product (str, optional): GOES product to download. Defaults to "ABI-L1b-Rad".
        satellite (int, optional): Satellite number. Defaults to 16.
        domain (str, optional): Domain of the data. Defaults to "F".
        save_dir (str, optional): Directory to save the downloaded data. Defaults to "./data".
        save_file_name (Optional[str], optional): Name of the file to save the resulting GeoDataFrame as a GeoJSON file.
            Defaults to None.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the downloaded GOES data.

    Raises:
        Exception: If an error occurs during the download process.

    Example:
        time_series = ["2022-01-01T00:00:00", "2022-01-01T00:10:00", "2022-01-01T00:20:00"]
        goes2go_download_from_timeseries(time_series, within="5 minutes", save_dir="./data", save_file_name="goes_data.csv")

    """
    goes_image_files = []

    # ABI Level 1b Data
    G16_DL = GOES(
        satellite=satellite,
        product=product,
        domain=domain,
    )
    # create within time
    within_freq, within_unit = within.split(" ")
    within = pd.to_timedelta(float(within_freq), unit=within_unit)

    pbar = tqdm(list(time_series))
    for itime in pbar:

        pbar.set_description(f"{itime}")
        try:
            idf_meta = G16_DL.nearesttime(
                attime=str(itime),
                within=within,
                return_as="filelist",
                save_dir=save_dir,
                download=True,
            )
            idf_meta["time"] = idf_meta["start"]

            goes_image_files.append(idf_meta)
        except Exception as e:
            raise Exception("An error occurred during the download process.") from e

    goes_image_files = pd.concat(goes_image_files, ignore_index=True)

    if save_file_name is not None:
        logger.info(f"Saving Meta-Information...")
        logger.debug(f"Save Path: {save_file_name}")
        Path(save_file_name).parent.mkdir(parents=True, exist_ok=True)
        goes_image_files.to_csv(
            save_file_name,
        )

    logger.info(f"Completed Query Script!")

    return goes_image_files


@app.command()
def goes2go_download_from_meta(
    meta_file_path: str = "./meta.csv",
    within: str = "10 minutes",
    product: str = "ABI-L1b-Rad",
    satellite: int = 16,
    domain: str = "F",
    save_dir: str = "./data",
    save_file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Downloads GOES data based on the metadata file.

    Args:
        meta_file_path (str, optional): Path to the metadata file. Defaults to "./meta.csv".
        within (str, optional): Time window for data retrieval. Defaults to "10 minutes".
        product (str, optional): GOES product to download. Defaults to "ABI-L1b-Rad".
        satellite (int, optional): Satellite number. Defaults to 16.
        domain (str, optional): Domain for data retrieval. Defaults to "F".
        save_dir (str, optional): Directory to save the downloaded data. Defaults to "./data".
        save_file_name (str, optional): Name of the file to save the downloaded data. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the downloaded GOES data.
    """
    # load file
    meta_file_path = Path(meta_file_path)
    
    assert meta_file_path.is_file()
    
    if meta_file_path.suffix == ".csv":
        df = pd.read_csv(meta_file_path)
    elif meta_file_path.suffix in [".geojson", ".json"]:
        df = gpd.read_file(meta_file_path)
    else:
        raise ValueError(f"Unrecognized filepath: {meta_file_path}")

    df_unique_times = df["time"].unique()

    unique_times = list(df_unique_times)
    
    return goes2go_download_from_timeseries(
        time_series=unique_times,
        within=within,
        product=product,
        satellite=satellite,
        domain=domain,
        save_dir=save_dir,
        save_file_name=save_file_name
    )


if __name__ == "__main__":
    app()
