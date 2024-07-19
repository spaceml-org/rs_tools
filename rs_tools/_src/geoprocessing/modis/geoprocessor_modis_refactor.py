import autoroot  # isort: skip
from dataclasses import dataclass
import datetime
from datetime import datetime
import os
from pathlib import Path
from typing import (
    List,
    Dict,
    Optional,
    Tuple,
    Union,
    Callable
)
import warnings
from georeader.griddata import footprint as swath_footprint
import dask
from georeader.griddata import footprint
from loguru import logger
import numpy as np
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import (
    BoundingBox,
    Geometry,
)

# GEOBOX
from odc.geo.xr import xr_coords
import pandas as pd
import geopandas as gpd
from pyresample import kd_tree
from pyresample.geometry import (
    GridDefinition,
    SwathDefinition,
)
import rioxarray
from satpy import Scene
from tqdm import tqdm, trange
import typer
import xarray as xr
from xrpatcher._src.base import XRDAPatcher
from rs_tools._src.geoprocessing.geometry import calculate_xrio_footprint


from rs_tools._src.data.modis import (
    MODIS_ID_TO_NAME,
    MODIS_NAME_TO_ID,
    MODISFileName,
    MODISRawFiles,
    get_modis_paired_files,
)
from rs_tools._src.data.modis.bands import get_modis_channel_numbers, CALIBRATION_CHANNELS
from rs_tools._src.data.modis.variables import VARIABLE_ATTRS
from rs_tools._src.geoprocessing.grid import create_latlon_grid
from rs_tools._src.geoprocessing.interp import resample_rioxarray
from rs_tools._src.geoprocessing.modis import (
    MODIS_WAVELENGTHS,
    format_modis_dates,
    parse_modis_dates_from_file,
)
from rs_tools._src.geoprocessing.modis.reproject import add_modis_crs
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.modis.reproject import regrid_swath_to_regular


dask.config.set(**{"array.slicing.split_large_chunks": False})
warnings.filterwarnings("ignore", category=FutureWarning)

app = typer.Typer()


def load_modis_data_raw(file: str, calibration: str="radiance") -> xr.Dataset:

    # load file with satpy
    scn = Scene(
        reader="modis_l1b",
        filenames=[
            str(file),
        ],
    )
    # get channels
    wishlist = CALIBRATION_CHANNELS[calibration]

    # load image
    scn.load(wishlist, generate=False, calibration=calibration)

    # convert to xarray
    ds = scn.to_xarray_dataset(datasets=wishlist)

    return ds


def preprocess_modis_raw(ds: xr.Dataset, calibration: str="radiance") -> xr.Dataset:
    """
    Preprocesses MODIS image radiances.

    Args:
        file (str): The file path of the MODIS image.

    Returns:
        xr.Dataset: The preprocessed MODIS image radiances as an xarray dataset.
    """

    # grab variable names from list of dataset variables
    channels = list(ds.data_vars)

    # Store the attributes in a dict before concatenation
    attrs_dict = {x: ds[x].attrs for x in channels}

    # concatinate in new band dimension, and defining a new variable name
    # NOTE: Concatination overwrites attrs of bands.
    ds = ds.assign({str(calibration): xr.concat(list(map(lambda x: ds[x], channels)), dim="band")})
    
    # drop duplicate variables
    ds = ds.drop_vars(list(map(lambda x: x, channels)))

    # ================
    # COORDINATES
    # ================
    # rename band dimensions
    ds = ds.assign_coords(band=list(map(lambda x: x, channels)))

    # convert measurement time (in seconds) to datetime
    time_stamp = pd.to_datetime(ds.attrs["start_time"])
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")
    # assign bands and time data to each variable
    ds = ds.assign_coords({"time": [time_stamp]})

    # assign band wavelengths
    ds = ds.assign_coords({"band_wavelength": list(MODIS_WAVELENGTHS.values())})

    # ================
    # ATTRIBUTES
    # ================

    # NOTE: Keep only certain relevant attributes
    ds.attrs = {}
    ds[calibration].attrs = {}
    ds[calibration].attrs = VARIABLE_ATTRS[calibration]
    ds[calibration].attrs["platform_name"] = attrs_dict[list(attrs_dict.keys())[0]]["platform_name"]
    ds[calibration].attrs["sensor"] = attrs_dict[list(attrs_dict.keys())[0]]["sensor"]

    # remove crs from dataset
    ds = ds.drop_vars("crs")

    return ds


def load_modis_cloud_mask_raw(file: str) -> xr.Dataset:
    # Load file using satpy scenes
    scn = Scene(
        reader="modis_l2",
        filenames=[
            str(file),
        ],
    )
    # Load cloud mask data
    datasets = scn.available_dataset_names()

    # Note: Needs to be loaded at 1000 m resolution for all channels to match
    scn.load(datasets, generate=False, resolution=1_000)

    # change to xarray data
    ds = scn.to_xarray_dataset()

    return ds


def combine_rads_and_clouds(ds_rads, ds_clouds):
    # extract cloud mask variable
    ds_clouds = ds_clouds["cloud_mask"]
    # save cloud mask as data coordinate
    ds_rads = ds_rads.assign_coords({"cloud_mask": (("y", "x"), ds_clouds.values)})
    # add cloud mask attrs to dataset
    ds_rads["cloud_mask"].attrs = ds_clouds.attrs

    # remove attrs that cause netcdf error
    for attr in ["start_time", "end_time", "area", "_satpy_id"]:
        ds_rads["cloud_mask"].attrs.pop(attr)

    # for var in ds_rads.data_vars:
    #     ds_rads[var].attrs = {}
    return ds_rads


@app.command()
def geoprocess_modis_aqua_terra(
        satellite_name: str = "aqua",
        calibration: str = "radiance",
        file_path: str = "./meta.geojson",
        save_path: str = "./",
        regridder: Optional[Callable] = False,
        ):
    """
    Geoprocesses MODIS Aqua/Terra data.

    Args:
        satellite_name (str, optional): Name of the satellite. Defaults to "aqua".
        calibration (str, optional): Calibration type. Defaults to "radiance".
        file_path (str, optional): Path to the meta-data file. Defaults to "./meta.geojson".
        save_path (str, optional): Path to save the processed data. Defaults to "./".
        regridder (Optional[Callable], optional): Function to regrid the data. Defaults to False.

    Returns:
        geopandas.DataFrame: GeoDataFrame containing the processed data.
    """
    # Initialize MODIS GeoProcessor
    logger.info(f"Initializing {satellite_name.upper()} GeoProcessor...")
    logger.debug(f"Path: {file_path}")
    # load meta-data
    gdf_meta = gpd.read_file(file_path)

    # get satellite ID
    logger.info(f"Loading Satellite Name: {satellite_name}")
    satellite_id = MODIS_NAME_TO_ID[satellite_name]
    logger.debug(f"Loading ID: {satellite_id}")
    # grab all files
    gdf_meta = gdf_meta.loc[(gdf_meta['satellite_id'] == satellite_id)]

    num_indexes = len(gdf_meta)

    logger.debug(f"Number of Files: {num_indexes}")

    pbar = tqdm(list(gdf_meta.iterrows()))

    geo_dateframes = list()
    for irow, idata in pbar:

        pbar.set_description(f"Satellite - {satellite_id} | Starting time: {idata['time']}...")

        # load modis swath
        pbar.set_description(f"Loading MODIS SWATH...")
        modis_swath = load_modis_data_raw(idata['file_path'], calibration=calibration)

        # preprocess modis swath
        pbar.set_description(f"Cleaning Data...")
        modis_swath = preprocess_modis_raw(modis_swath, calibration=calibration).compute()


        if regridder is not None:
            pbar.set_description(f"Regridding from SWATH to Regular...")
            modis_swath = regridder(modis_swath, variable=calibration)

        # check if save path exists, and create if not
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # remove file if it already exists
        itime_name = datetime.strftime(idata['time'], "%Y%m%d%H%M%S")
        if regridder is not None:
            save_filename = f"{itime_name}_{satellite_name}_{calibration}_grid.nc"
        else:
            save_filename = f"{itime_name}_{satellite_name}_{calibration}_swath.nc"

        full_path = Path(save_path).joinpath(save_filename)
        if os.path.exists(full_path):
            logger.info(f"File already exists. Overwriting file: {full_path}")
            os.remove(full_path)

        modis_swath.rio.write_crs("epsg:4326", inplace=True)

        modis_swath.to_netcdf(full_path, engine="netcdf4")

        # create pandas geodataframe
        df = pd.DataFrame({
            "time": idata["time"],
            "satellite_id": MODIS_NAME_TO_ID[satellite_name],
            "satellite_name": satellite_name,
            "filename": str(save_filename),
            "full_path": str(full_path),
            "cloud_mask": False,
            }, index=[0])
        

        if regridder:
            # use absolute swath footprint
            modis_polygons = calculate_xrio_footprint(modis_swath)
        else:

            modis_polygons = swath_footprint(modis_swath.longitude.values, modis_swath.latitude.values)


        # create a GeoDataFrame 
        geometry = gpd.GeoSeries([modis_polygons], crs=modis_swath.rio.crs)

        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=modis_swath.rio.crs)

        geo_dateframes.append(gdf)
        
    logger.info(f"Creating Meta data file...")
    geo_dateframes = pd.concat(geo_dateframes, ignore_index=True)

    # save to geojson file
    logger.info(f"Saving Meta-Information...")
    
    if regridder is not None:
        save_filename = Path(f"{satellite_name}_{calibration}_grid_meta.geojson")
    else:
        save_filename = Path(f"{satellite_name}_{calibration}_swath_meta.geojson")

    full_path = Path(save_path).joinpath(save_filename)
    logger.debug(f"Save Path: {full_path}")
    geo_dateframes.to_file(full_path, driver="GeoJSON")

    return geo_dateframes


@app.command()
def geoprocess_modis_aqua_terra_clouds(
        satellite_name: str = "aqua", 
        calibration: str = "radiance", 
        file_path: str = "./meta.geojson", 
        save_path: str = "./",
        regridder: Optional[Callable]=False,
        save_file_type: str="zarr",
        ):
    """
    Geoprocesses MODIS Aqua and Terra satellite data for cloud analysis.

    Args:
        satellite_name (str, optional): Name of the satellite. Defaults to "aqua".
        calibration (str, optional): Calibration type. Defaults to "radiance".
        file_path (str, optional): Path to the meta-data file. Defaults to "./meta.geojson".
        save_path (str, optional): Path to save the processed data. Defaults to "./".
        regridder (Optional[Callable], optional): Function for regridding. Defaults to False.

    Returns:
        pd.DataFrame: GeoDataFrame containing the processed data.
    """
    # Initialize MODIS GeoProcessor
    logger.info(f"Initializing {satellite_name.upper()} GeoProcessor...")
    logger.debug(f"Path: {file_path}")
    # load meta-data
    gdf_meta = gpd.read_file(file_path)

    # get satellite ID
    logger.info(f"Loading Satellite Name: {satellite_name}")
    satellite_id = MODIS_NAME_TO_ID[satellite_name]
    logger.debug(f"Loading ID: {satellite_id}")
    # get cloud satellite ID
    logger.info(f"Loading Satellite Name: {satellite_name}_cloud")
    satellite_id_cloud = MODIS_NAME_TO_ID[f"{satellite_name}_cloud"]
    logger.debug(f"Loading ID: {satellite_id_cloud}")
    # grab all files
    gdf_meta = gdf_meta[gdf_meta["satellite_id"].isin([satellite_id, satellite_id_cloud])]

    logger.info(f"Number of Files: {len(gdf_meta)}")

    # groupby unique times
    itimes = gdf_meta["time"].unique()
    logger.info(f"Number of Times: {len(itimes)}")

    pbar = tqdm(itimes)

    geo_dateframes = list()
    for itime in pbar:
        pbar.set_description(f"Satellite - {satellite_name} | Starting time: {itime}...")

        # extract satellites
        igdf_meta = gdf_meta.loc[gdf_meta["time"] == itime]
        # print(igdf_meta)

        # extract the appropriate satellites
        igdf = igdf_meta.loc[igdf_meta["satellite_id"] ==  satellite_id].iloc[0]
        igdf_clouds = igdf_meta.loc[igdf_meta["satellite_id"] ==  satellite_id_cloud].iloc[0]

        # print(igdf["file_path"].iloc[0])
        # print(igdf_clouds["file_path"].iloc[0])

        # load modis swath
        pbar.set_description(f"Loading MODIS SWATH...")
        modis_swath = load_modis_data_raw(igdf["file_path"], calibration=calibration)

        # preprocess modis swath
        pbar.set_description(f"Cleaning Data...")
        modis_swath = preprocess_modis_raw(modis_swath, calibration=calibration).compute()

        pbar.set_description(f"Loading Cloud Mask File...")
        cloud_mask = load_modis_cloud_mask_raw(igdf_clouds["file_path"])

        pbar.set_description(f"Combining files...")
        modis_swath = combine_rads_and_clouds(modis_swath, cloud_mask)

        
        if regridder is not None:
            pbar.set_description(f"Regridding from SWATH to Regular...")
            
            modis_swath_calib = regridder(modis_swath, variable=calibration)

            modis_swath_clouds = regridder(modis_swath, variable="cloud_mask")

            modis_swath = modis_swath_calib

            modis_swath["cloud_mask"] = modis_swath_clouds["cloud_mask"]
        else:
            cloud_mask = modis_swath["cloud_mask"]
            modis_swath = modis_swath.drop_vars(["cloud_mask"])
            modis_swath["cloud_mask"] = cloud_mask
            
        modis_swath.rio.write_crs("epsg:4326", inplace=True)

        # check if save path exists, and create if not
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # remove file if it already exists
        itime_name = datetime.strftime(itime, "%Y%m%d%H%M%S")
        if regridder is not None:
            save_filename = Path(f"{itime_name}_{satellite_name}_{calibration}_grid.nc")
        else:
            save_filename = Path(f"{itime_name}_{satellite_name}_{calibration}_swath.nc")
        if os.path.exists(save_filename):
            logger.info(f"File already exists. Overwriting file: {save_filename}")
            os.remove(save_filename)


        full_path = Path(save_path).joinpath(save_filename)
        modis_swath.to_netcdf(full_path, engine="netcdf4")

        # create pandas geodataframe
        df = pd.DataFrame({
            "time": itime,
            "satellite_id": MODIS_NAME_TO_ID[satellite_name],
            "satellite_name": satellite_name,
            "filename": str(save_filename),
            "full_path": str(full_path),
            "cloud_mask": True,
            }, index=[0])
        

        if regridder:
            # use absolute swath footprint
            modis_polygons = calculate_xrio_footprint(modis_swath)
        else:

            modis_polygons = swath_footprint(modis_swath.longitude.values, modis_swath.latitude.values)

        # create a GeoDataFrame 
        geometry = gpd.GeoSeries([modis_polygons], crs=modis_swath.rio.crs)

        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=modis_swath.rio.crs)

        geo_dateframes.append(gdf)
        
        
    logger.info(f"Creating Meta data file...")
    geo_dateframes = pd.concat(geo_dateframes, ignore_index=True)

    # save to geojson file
    logger.info(f"Saving Meta-Information...")
    
    if regridder is not None:
        save_filename = Path(f"{satellite_name}_{calibration}_grid_meta.geojson")
    else:
        save_filename = Path(f"{satellite_name}_{calibration}_swath_meta.geojson")

    full_path = Path(save_path).joinpath(save_filename)
    logger.debug(f"Save Path: {full_path}")
    geo_dateframes.to_file(full_path, driver="GeoJSON", )

    return geo_dateframes


@app.command()
def geoprocess_aqua_terra(
    satellite_name: str = "aqua",
    calibration: str="radiance",
    file_path: str = "./meta.geojson",
    save_path: str = "./",
    regridder: Optional[Callable] = None,
    cloud_mask: bool=False
):
    """
    Geoprocesses MODIS files

    Args:
        satellite_name (str, optional): The name of the satellite. Valid values are "aqua" and "terra". Defaults to "aqua".
        calibration (str, optional): The calibration type. Defaults to "radiance". Options: "radiance", "reflectance", "counts"
        read_path (str, optional): The path to read the files from. Defaults to "./".
        save_path (str, optional): The path to save the geoprocessed files to. Defaults to "./".

    Returns:
        None

    Raises:
        ValueError: If the satellite name is not recognized.

    Notes:
    -----
    - This function geoprocesses MODIS files based on the specified satellite and calibration type.
    - Currently, only "aqua" and "terra" satellites are supported.
    - The geoprocessing is performed on the files located in the read_path directory.
    - The geoprocessed files are saved in the save_path directory.
    """


    if satellite_name in ["aqua", "terra"]:
        assert calibration in ["radiance", "reflectance", "counts", "brightness_temperature"]
        if not cloud_mask:
            geoprocess_modis_aqua_terra(
                satellite_name,
                calibration=calibration,
                file_path=file_path,
                save_path=save_path,
                regridder=regridder,
                )
        else:
            geoprocess_modis_aqua_terra_clouds(
                satellite_name,
                calibration=calibration,
                file_path=file_path,
                save_path=save_path,
                regridder=regridder,
                )
    else:
        raise ValueError(f"Unrecognized satellite name: {satellite_name}")

    logger.info(f"GeoProcessing Files...")
    logger.info(f"Finished {satellite_name.upper()} GeoProcessing Script...!")


if __name__ == "__main__":
    """
    # =========================
    # Test Cases
    # =========================
    python geoprocessor_modis_refactor.py --satellite aqua --read-path "/home/data" --save-path /home/data/modis/geoprocessed
    python geoprocessor_modis_refactor.py --satellite terra --read-path "/home/data" --save-path /home/data/modis/geoprocessed

    # =========================
    # FAILURE TEST CASES
    # =========================
    """
    typer.run(geoprocess_aqua_terra)
