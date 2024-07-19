import autoroot
from typing import Optional, Tuple, List, Union
from tqdm.auto import tqdm
import typer
from pathlib import Path
import pandas as pd
import geopandas as gpd
from odc.geo.geom import BoundingBox
from rs_tools._src.data.ea.query import query_ea_timestamps, ea_granule_to_gdf
from loguru import logger
import earthaccess
from rs_tools._src.utils.io import get_list_filenames

app = typer.Typer()




@app.command()
def ea_download_from_query(
    file_path: str="./",
    save_path: str="./",
    meta_save_file_name: str="meta"
) -> gpd.GeoDataFrame:
    """
    Downloads MODIS data based on a query.

    Args:
        file_path (str, optional): Path to the query file. Defaults to "./".
        save_path (str, optional): Path to save the downloaded data. Defaults to "./".

    Raises:
        ValueError: If the file_path is of an unrecognized filetype.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the downloaded MODIS data.
    """
    logger.info(f"Gathering EA query...")
    if isinstance(file_path, str):
        logger.info(f"Loading EA file...")
        ea_query = gpd.read_file(file_path)
    elif isinstance(file_path, gpd.GeoDataFrame):
        ea_query = file_path
    else:
        msg = f"Unrecognized filetype: {type(file_path)}"
        raise ValueError(msg)
        
    # iterate through rows of dataframe
    pbar_query = tqdm(list(ea_query.iterrows()))

    geo_dateframes = list()

    for irow, ifile in pbar_query:

        pbar_query.set_description(f"Downloading: {ifile.satellite_id} | Time: {ifile.time}")

        ea_granules = query_ea_timestamps(
            short_name=ifile.satellite_id,
            bounding_box=ifile.geometry.bounds,
            temporal=(ifile.time, ifile.time)
        )

        earthaccess.download(ea_granules, save_path)

        # create granule
        gdf = ea_granule_to_gdf(ea_granules)

        geo_dateframes.append(gdf)

    geo_dateframes = pd.concat(geo_dateframes, ignore_index=True).drop_duplicates()

    # list of all .hdf files in the directory
    all_files = get_list_filenames(save_path, ".hdf")


    for irow, imeta in geo_dateframes.iterrows():
        
        # search for satellite_id & time
        # grab dates
        file_query = list(filter(lambda x: imeta["filename"] in x, all_files))
        file_query = list(set(file_query))
        if not file_query:
            pass
        else:
            file_query = list(set(file_query))[0]
            geo_dateframes.loc[irow, "file_path"] = str(Path(file_query))

    
    # clean the dataframe
    geo_dateframes = geo_dateframes.drop_duplicates().reset_index(drop=True)
    
    # save to geojson file
    logger.info(f"Saving Meta-Information...")
    save_path = Path(save_path).joinpath(f"{meta_save_file_name}.geojson")
    logger.debug(f"Save Path: {save_path}")
    geo_dateframes.to_file(save_path, driver="GeoJSON")

    return geo_dateframes



if __name__ == '__main__':
    app()