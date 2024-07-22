from typing import Optional, List
import typer
from pathlib import Path
from odc.geo.geom import BoundingBox, Geometry
from rs_tools._src.data.msg.meta import _check_eumdac_login, MSGGranule, granule_to_dataframe
import eumdac
from eumdac.product import Product
from datetime import datetime
import pandas as pd
import geopandas as gpd
from pandas._libs.tslibs.timedeltas import Timedelta
from pandas._libs.tslibs.timestamps import Timestamp
from rs_tools._src.preprocessing.periods import calculate_nearest_timestamp_index
from rs_tools._src.preprocessing.periods import TimeQuery, TimeSeries
from tqdm import tqdm


app = typer.Typer()


def msg_query_collection_time_bounds(
    collection_id: str,
    time_start: Timestamp,
    time_end: Timestamp,
    bounding_box: Optional[BoundingBox] = None,
    eumdac_key: str = "",
    eumdac_secret: str = "",
    meta_save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query the MSG collection for the nearest granule to a given timestamp.

    Args:
        collection_id (str): The ID of the collection to query.
        time_stamp (Timestamp): The target timestamp.
        within (Timedelta): The time range within which to search for the nearest granule.
        bounding_box (Optional[BoundingBox], optional): The bounding box to filter the search. Defaults to None.
        eumdac_key (str, optional): The EUMDAC API key. Defaults to "".
        eumdac_secret (str, optional): The EUMDAC API secret. Defaults to "".

    Returns:
        MSGGranule: The nearest MSG granule to the target timestamp.
    """
    if isinstance(time_start, str):
        time_start = pd.to_datetime(time_start)
    if isinstance(time_end, str):
        time_end = pd.to_timedelta(time_end)
        
    # grab token
    token = _check_eumdac_login(eumdac_key=eumdac_key, eumdac_secret=eumdac_secret)
    # initialize datastore
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection_id)
    if bounding_box is not None:
        raise NotImplementedError(f"Bounding box not implemented...")
    #     # geo = 'POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in bounding_box.points]))
    
        
    products = selected_collection.search(
        bbox=bounding_box,
        dtstart=time_start,
        dtend=time_end,
    )
    
    msg_granules = list()
    for iproduct in products:
        msg_granules.append(MSGGranule(iproduct))
        
    
    # convery to dataframed 
    dfs = list(map(lambda x: granule_to_dataframe(x), msg_granules))
    
    # concat all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # clean dataframe
    df = df.drop_duplicates().reset_index(drop=True)
    
    # sort and drop (just and cases)

    if meta_save_path:
        Path(meta_save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(meta_save_path)
    
    return df


def msg_query_collection_nearest(
    collection_id: str,
    time_stamp: Timestamp,
    within: Timedelta,
    bounding_box: Optional[BoundingBox] = None,
    eumdac_key: str = "",
    eumdac_secret: str = "",
) -> pd.DataFrame:
    """
    Query the MSG collection for the nearest granule to a given timestamp.

    Args:
        collection_id (str): The ID of the collection to query.
        time_stamp (Timestamp): The target timestamp.
        within (Timedelta): The time range within which to search for the nearest granule.
        bounding_box (Optional[BoundingBox], optional): The bounding box to filter the search. Defaults to None.
        eumdac_key (str, optional): The EUMDAC API key. Defaults to "".
        eumdac_secret (str, optional): The EUMDAC API secret. Defaults to "".

    Returns:
        MSGGranule: The nearest MSG granule to the target timestamp.
    """
    if isinstance(time_stamp, str):
        time_stamp = pd.to_datetime(time_stamp)
    if isinstance(within, str):
        within = pd.to_timedelta(within)
        
    time_start = time_stamp - within
    time_end = time_stamp + within
    
    # grab token
    token = _check_eumdac_login(eumdac_key=eumdac_key, eumdac_secret=eumdac_secret)
    # initialize datastore
    datastore = eumdac.DataStore(token)
    selected_collection = datastore.get_collection(collection_id)
    if bounding_box is not None:
        raise NotImplementedError(f"Bounding box not implemented...")
    #     # geo = 'POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in bounding_box.points]))
    
        
    products = selected_collection.search(
        bbox=bounding_box,
        dtstart=time_start,
        dtend=time_end,
    )
    
    msg_granules = list()
    for iproduct in products:
        msg_granules.append(MSGGranule(iproduct))
        
    # create list of timestamps
    time_stamps = pd.to_datetime(list(map(lambda x: x.datetime_acquisition, msg_granules)))
    
    # get the nearest index
    nearest_index = calculate_nearest_timestamp_index(time_stamps, time_stamp)
    
    # return nearest time stamp
    msg_granule = msg_granules[nearest_index]
    
    # conver to dataframe
    df = granule_to_dataframe(msg_granule)
    
    # add query time
    df["time_query"] = time_stamp
    
    return df


def msg_query_product(
    product_id: str,
    collection_id: str,
    eumdac_key: str ="",
    eumdac_secret: str="",
    # format: str="geotiff"
):
    
    # grab token
    token = _check_eumdac_login(eumdac_key=eumdac_key, eumdac_secret=eumdac_secret)
    # initialize datastore
    datastore = eumdac.DataStore(token)
    datatailor = eumdac.DataTailor(token)
    
    # selected product
    selected_product = datastore.get_product(
        collection_id=collection_id,
        product_id=product_id
    )
    # chain = eumdac.tailor_models.Chain(
    #     product=product_id.split(":")[-1],
    #     format="geotiff"
    # )
    
    # customisation = datatailor.new_customisation(selected_product, chain)
    
    return MSGGranule(selected_product)
    

def msg_query_collection_timeseries(
    time_series: List[Timestamp],
    collection_id: str,
    within: str = "15 minutes",
    bounding_box: Optional[BoundingBox] = None,
    eumdac_key: str = "",
    eumdac_secret: str = "",
    meta_save_path: Optional[str]=None,
):
    
    dataframes = list()
    
    pbar = tqdm(list(time_series), leave=False)
    for itime in pbar:

        pbar.set_description(f"{itime}")
        idf = msg_query_collection_nearest(
            time_stamp=itime,
            within=within,
            collection_id=collection_id,
            bounding_box=bounding_box,
            eumdac_key=eumdac_key,
            eumdac_secret=eumdac_secret
        )
        idf["time_query"] = itime
        dataframes.append(idf)
    
    # concatenate all dateframes    
    dataframes = pd.concat(dataframes, ignore_index=True)
    
    # sort and drop (just and cases)
    dataframes = dataframes.drop_duplicates().reset_index(drop=True)
    if meta_save_path:
        Path(meta_save_path).parent.mkdir(parents=True, exist_ok=True)
        dataframes.to_csv(meta_save_path)
    
    return dataframes    


def msg_query_collection_meta(
    meta_load_path: str,
    collection_id: str,
    column_name: str="time",
    within: str = "15 minutes",
    bounding_box: Optional[BoundingBox] = None,
    eumdac_key: str = "",
    eumdac_secret: str = "",
    meta_save_path: Optional[str]=None,
):
    # load meta file
    # load file
    if isinstance(meta_load_path, pd.DataFrame | gpd.GeoDataFrame):
        df = meta_load_path
        
    elif isinstance(meta_load_path, Path | str):
        
        meta_load_path = Path(meta_load_path)
        
        assert meta_load_path.is_file()
        
        if meta_load_path.suffix == ".csv":
            df = pd.read_csv(meta_load_path, index_col=0)
            
        elif meta_load_path.suffix in [".geojson", ".json"]:
            df = gpd.read_file(meta_load_path, index_col=0)
        else:
            raise ValueError(f"Unrecognized filepath extension: {meta_load_path.suffix}")
    else:
        msg = f"Unrecognized filetype: {meta_load_path}"
        raise ValueError(msg)
    
    time_series = df[column_name]
    
    return msg_query_collection_timeseries(
        time_series=list(time_series),
        within=within,
        collection_id=collection_id,
        bounding_box=bounding_box,
        meta_save_path=meta_save_path,
        eumdac_key=eumdac_key,
        eumdac_secret=eumdac_secret
    )


def msg_query_collection_timequery(
    collection_id: str,
    time_query: TimeQuery,
    bounding_box: Optional[BoundingBox] = None,
    eumdac_key: str = "",
    eumdac_secret: str = "",
) -> pd.DataFrame:
    """
    Query the MSG collection for the nearest granule to a given timestamp.

    Args:
        collection_id (str): The ID of the collection to query.
        time_stamp (Timestamp): The target timestamp.
        within (Timedelta): The time range within which to search for the nearest granule.
        bounding_box (Optional[BoundingBox], optional): The bounding box to filter the search. Defaults to None.
        eumdac_key (str, optional): The EUMDAC API key. Defaults to "".
        eumdac_secret (str, optional): The EUMDAC API secret. Defaults to "".

    Returns:
        MSGGranule: The nearest MSG granule to the target timestamp.
    """
    
    return msg_query_collection_nearest(
        collection_id=collection_id,
        time_stamp=time_query.time_stamp,
        within=time_query.time_window,
        bounding_box=bounding_box,
        eumdac_key=eumdac_key,
        eumdac_secret=eumdac_secret
    )


if __name__ == '__main__':
    app()

