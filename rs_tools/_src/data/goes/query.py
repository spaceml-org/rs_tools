
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




@app.command()
def goes2go_data_query_bounds(
    time_start: str="2020-10-01",
    time_end: str="2020-10-02",
    product: str="ABI-L1b-Rad",
    satellite: int=16,
    domain: str="F",
    save_file_name: Optional[str]=None
) -> gpd.GeoDataFrame:
    """
    Query MODIS data for the specified satellite names, bounding box, and date range.
    
    Args:
        satellite_ids (List[str], optional): List of satellite IDs. Defaults to ["MYD021KM"].
        bbox_string (str, optional): Bounding box coordinates in the format "lon_min lat_min lon_max lat_max". 
            Defaults to "-180 -90 180 90".
        start_date (str, optional): Start date in the format "YYYY-MM-DD". Defaults to "2020-10-01".
        end_date (str, optional): End date in the format "YYYY-MM-DD". Defaults to "2020-10-02".
        save_file_name (Optional[str], optional): Name of the file to save the resulting GeoDataFrame as a GeoJSON file. 
            Defaults to None.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing MODIS data for the specified parameters.
    """
    # ABI Level 1b Data
    G16_DL = GOES(
        satellite=satellite, 
        product=product, 
        domain=domain,
    )

    img_meta = G16_DL.timerange(
        start=time_start,
        end=time_end,
        return_as="filelist",
        save_dir=None,
        download=False
    )

    logger.info(f"Found {len(img_meta)} Total Files...")


    # save to geojson file
    
    
    if save_file_name is not None:
        logger.info(f"Saving Meta-Information...")
        logger.debug(f"Save Path: {save_file_name}")
        Path(save_file_name).parent.mkdir(parents=True, exist_ok=True)
        img_meta.to_csv(save_file_name, )

    logger.info(f"Completed Query Script!")
    return img_meta


def goes2go_data_query_timeseries(
    time_series: TimeSeries,
    product: str="ABI-L1b-Rad",
    satellite: int=16,
    domain: str="F",
    save_file_name: Optional[str]=None
    ) -> gpd.GeoDataFrame:
    """
    Query MODIS data for the specified satellite names, bounding box, and date range.
    å
    Args:
        satellite_ids (List[str], optional): List of satellite IDs. Defaults to ["MYD021KM"].
        bbox_string (str, optional): Bounding box coordinates in the format "lon_min lat_min lon_max lat_max". 
            Defaults to "-180 -90 180 90".
        start_date (str, optional): Start date in the format "YYYY-MM-DD". Defaults to "2020-10-01".
        end_date (str, optional): End date in the format "YYYY-MM-DD". Defaults to "2020-10-02".
        save_file_name (Optional[str], optional): Name of the file to save the resulting GeoDataFrame as a GeoJSON file. 
            Defaults to None.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing MODIS data for the specified parameters.
    """

    # ABI Level 1b Data
    G16_DL = GOES(
        satellite=satellite, 
        product=product, 
        domain=domain,
    )



    goes_image_files = list()

    pbar = tqdm(list(time_series))
    for itime in pbar:

        pbar.set_description(f"{itime.time_stamp}")
        try:
            idf_meta = G16_DL.nearesttime(
                    attime=itime.time_stamp,
                    within=itime.time_window,
                    return_as="filelist",
                    save_dir=None,
                    download=False
            )
            idf_meta["time"] = idf_meta["start"]
            idf_meta["time_query"] = itime.time_stamp

            goes_image_files.append(
                idf_meta
            )
        # TODO: put a concrete error here
        except FileNotFoundError:
            pass

    if len(goes_image_files) > 0:
        goes_image_files = pd.concat(goes_image_files, ignore_index=True)

        goes_image_files = goes_image_files.drop_duplicates()

        if save_file_name is not None:
            logger.info(f"Saving Meta-Information...")
            logger.debug(f"Save Path: {save_file_name}")
            Path(save_file_name).parent.mkdir(parents=True, exist_ok=True)
            goes_image_files.to_csv(save_file_name, )

    logger.info(f"Completed Query Script!")

    return goes_image_files


@app.command()
def goes2go_data_query_meta(
    meta_file: str="./meta.geojson",
    within: str="10 minutes",
    product: str="ABI-L1b-Rad",
    satellite: int=16,
    domain: str="F",
    save_file_name: Optional[str]=None
    ) -> gpd.GeoDataFrame:
    """
    Query MODIS data for the specified satellite names, bounding box, and date range.
    å
    Args:
        satellite_ids (List[str], optional): List of satellite IDs. Defaults to ["MYD021KM"].
        bbox_string (str, optional): Bounding box coordinates in the format "lon_min lat_min lon_max lat_max". 
            Defaults to "-180 -90 180 90".
        start_date (str, optional): Start date in the format "YYYY-MM-DD". Defaults to "2020-10-01".
        end_date (str, optional): End date in the format "YYYY-MM-DD". Defaults to "2020-10-02".
        save_file_name (Optional[str], optional): Name of the file to save the resulting GeoDataFrame as a GeoJSON file. 
            Defaults to None.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing MODIS data for the specified parameters.
    """

    # ABI Level 1b Data
    G16_DL = GOES(
        satellite=satellite, 
        product=product, 
        domain=domain,
    )
    within_freq, within_unit = within.split(" ")
    within = pd.to_timedelta(float(within_freq), unit=within_unit)

    logger.info(f"Within: {within}")

    # open file
    gdf = gpd.read_file(meta_file)

    goes_image_files = list()

    pbar = tqdm(list(gdf.iterrows()))
    for irow, imeta in pbar:

        pbar.set_description(f"{imeta['time']}")
        try:
            idf_meta = G16_DL.nearesttime(
                    attime=str(imeta["time"]),
                    within=within,
                    return_as="filelist",
                    save_dir=None,
                    download=False
            )
            idf_meta["time"] = idf_meta["start"]
            idf_meta["time_query"] = imeta["time"]

            goes_image_files.append(
                idf_meta
            )
        # TODO: put a concrete error here
        except:
            pass

    goes_image_files = pd.concat(goes_image_files, ignore_index=True)
    

    if save_file_name is not None:
        logger.info(f"Saving Meta-Information...")
        logger.debug(f"Save Path: {save_file_name}")
        Path(save_file_name).parent.mkdir(parents=True, exist_ok=True)
        goes_image_files.to_csv(save_file_name, )

    logger.info(f"Completed Query Script!")

    return goes_image_files

    # # ABI Level 1b Data
    # G16_DL = GOES(
    #     satellite=satellite, 
    #     product=product, 
    #     domain=domain,
    # )

    # img_meta = G16_DL.timerange(
    #     start=time_start,
    #     end=time_end,
    #     return_as="filelist",
    #     save_dir=None,
    #     download=False
    # )

    # logger.info(f"Found {len(img_meta)} Total Files...")


    # # save to geojson file
    
    
    # if save_file_name is not None:
    #     logger.info(f"Saving Meta-Information...")
    #     logger.debug(f"Save Path: {save_file_name}")
    #     Path(save_file_name).parent.mkdir(parents=True, exist_ok=True)
    #     img_meta.to_csv(save_file_name, )

    # logger.info(f"Completed Query Script!")
    # return img_meta

if __name__ == '__main__':
    app()