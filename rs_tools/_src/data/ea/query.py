
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
from earthaccess.results import DataGranule
from datetime import datetime
from odc.geo.geom import Geometry, polygon
import geopandas as gpd
import pandas as pd
from rs_tools._src.geoprocessing.geometry import bbox_string_to_bbox
from dataclasses import dataclass
from rs_tools._src.data.ea.meta import ea_granule_to_gdf

app = typer.Typer()


def query_ea_timestamps(
        short_name: str,
        bounding_box: tuple,
        temporal: tuple,
        ):
    """
    Function to query the Earthdata API for MODIS data timestamps.
    
    :param short_name: MODIS short name (e.g. 'MOD021KM' for Terra MODIS Level 1B data at 1km resolution)
    :param bounding_box: The region to be queried. Follows format (min_lon, min_lat, max_lon, max_lat).
    :param temporal: Min and max date/time to be queried. Follows format (start_datetime: YYYY-MM-DD HH:MM:SS, end_datetime: YYYY-MM-DD HH:MM:SS).
    
    :return: result object
    """
    results = earthaccess.search_data(
        short_name=short_name,
        cloud_hosted=True,
        bounding_box=bounding_box,
        temporal=temporal,
        count=-1
    )
    return results


@app.command()
def ea_data_query(
        satellite_ids: List[str] = ["MYD021KM"],
        bbox_string: str = "-180 -90 180 90",
        start_date: str = "2020-10-01",
        end_date: str = "2020-10-02",
        save_file_name: Optional[str] = None
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
    if isinstance(satellite_ids, str):
        satellite_ids = [satellite_ids]
    # create bounding box from string
    bbox = bbox_string_to_bbox(bbox_string=bbox_string)

    logger.debug(f"Bounding Box: {bbox.bbox}")
    logger.debug(f"Date Range: {start_date} - {end_date}")

    geo_dataframes = list()

    for isatellite_id in satellite_ids:
        

        logger.debug(f"Satellite ID: {isatellite_id}")
        # TODO: validate date-range
        # save granules to file
        ea_granules = query_ea_timestamps(
            short_name=isatellite_id,
            bounding_box=bbox.bbox,
            temporal=(start_date, end_date)
        )
        
        logger.info(f"Found {len(ea_granules)} MODIS granules for field-of-view and specified time period...")

        # create granule
        gdf = ea_granule_to_gdf(ea_granules)

        geo_dataframes.append(gdf)

    geo_dataframes = pd.concat(geo_dataframes, ignore_index=True)

    # clean the dataframe
    geo_dataframes = geo_dataframes.drop_duplicates().reset_index(drop=True)
    
    # save to geojson file
    logger.info(f"Saving Meta-Information...")
    logger.debug(f"Save Path: {save_file_name}")
    if save_file_name is not None:
        Path(save_file_name).parent.mkdir(parents=True, exist_ok=True)
        geo_dataframes.to_file(save_file_name, driver="GeoJSON", )

    logger.info(f"Completed Query Script!")
    
    return geo_dataframes


if __name__ == '__main__':
    app()