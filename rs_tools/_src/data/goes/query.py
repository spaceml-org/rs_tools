
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

app = typer.Typer()


def query_ea_timestamps(
        short_name: str,
        bounding_box: tuple,
        temporal: tuple,
        ):
    """
    Function to query the Earthdata API for MODIS data timestamps.
    :param short_name: MODIS short name (e.g. 'MOD021KM' for Terra MODIS Level 1B data at 1km resolution)
    :param bounding_box (tuple, optional): The region to be queried. Follows format (min_lon, min_lat, max_lon, max_lat).
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


def ea_granule_to_datetime(granule: earthaccess.results.DataGranule) -> datetime:
    """
    Function to convert a MODIS granule to a datetime object.
    :param granule: MODIS granule
    :return: datetime object
    """
    return datetime.strptime(granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")


def ea_granule_to_polygon(granule: earthaccess.results.DataGranule) -> Geometry:
    """
    Converts a MODIS granule to a polygon geometry.

    Args:
        granule (earthaccess.results.DataGranule): The MODIS granule to convert.

    Returns:
        Geometry: The polygon geometry representing the MODIS granule.
    """

    points_list = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
    coordinates = [(point['Longitude'], point['Latitude']) for point in points_list]
    return polygon(coordinates, crs="4326")


def ea_granule_to_satellite_id(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["CollectionReference"]["ShortName"]


def ea_granule_to_filename(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]

def ea_granule_to_satellite_name(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["Platforms"][0]["ShortName"].lower()


def ea_granule_to_instrument(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["Platforms"][0]["Instruments"][0]["ShortName"]


def ea_granule_to_daynightflay(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["DataGranule"]["DayNightFlag"]


def ea_granule_to_gdf(ea_granules):

    # gather all modis timestamps
    ea_timestamps = list(map(ea_granule_to_datetime, ea_granules))

    # gather all polygons
    ea_polygons = list(map(ea_granule_to_polygon, ea_granules))

    # gather satellite ids
    ea_satellite_ids = list(map(ea_granule_to_satellite_id, ea_granules))

    # gather satellite names
    ea_satellite_names = list(map(ea_granule_to_satellite_name, ea_granules))

    # gather satellite filenames
    ea_satellite_filenames = list(map(ea_granule_to_filename, ea_granules))

    # gather satellite filenames
    ea_satellite_instruments = list(map(ea_granule_to_instrument, ea_granules))

    # day
    ea_satellite_daynightflag = list(map(ea_granule_to_daynightflay, ea_granules))

    # create pandas geodataframe
    df = pd.DataFrame({
        "time": ea_timestamps,
        "satellite_id": ea_satellite_ids,
        "satellite_name": ea_satellite_names,
        "filename": ea_satellite_filenames,
        "daynight": ea_satellite_daynightflag,
        "satellite_instrument": ea_satellite_instruments,
        })

    # convert the list of polygons into a GeoSeries
    geometry = gpd.GeoSeries(ea_polygons)

    # create a GeoDataFrame 
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    return gdf




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


@app.command()
def goes2go_data_query_meta(
    meta_file: str="./",
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

        print(imeta["time"])
        goes_image_files.append(G16_DL.nearesttime(
                attime=str(imeta["time"]),
                within=within,
                return_as="filelist",
                save_dir=None,
                download=False
                )
        )

    goes_image_files = pd.concat(goes_image_files, ignore_index=True)

    if save_file_name is not None:
        logger.info(f"Saving Meta-Information...")
        logger.debug(f"Save Path: {save_file_name}")
        Path(save_file_name).parent.mkdir(parents=True, exist_ok=True)
        goes_image_files.to_csv(save_file_name, )

    logger.info(f"Completed Query Script!")

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