import autoroot  # isort: skip

from dataclasses import dataclass
import datetime
from datetime import datetime
import os
from pathlib import Path
from typing import (
    List,
    Optional,
    Tuple,
    Union,
)
import warnings

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
from pyresample import kd_tree
from pyresample.geometry import (
    GridDefinition,
    SwathDefinition,
)
import rioxarray
from satpy import Scene
from tqdm import tqdm
import typer
import xarray as xr
from xrpatcher._src.base import XRDAPatcher

from rs_tools import (
    MODIS_VARIABLES,
    get_modis_channel_numbers,
    modis_download,
)
from rs_tools._src.data.modis import (
    MODIS_ID_TO_NAME,
    MODIS_NAME_TO_ID,
    MODISFileName,
    MODISRawFiles,
    get_modis_paired_files,
)
from rs_tools.modis import get_modis_channel_numbers, CALIBRATION_CHANNELS, VARIABLE_ATTRS
from rs_tools._src.geoprocessing.grid import create_latlon_grid
from rs_tools._src.geoprocessing.interp import resample_rioxarray
from rs_tools._src.geoprocessing.modis import (
    MODIS_WAVELENGTHS,
    format_modis_dates,
    parse_modis_dates_from_file,
)
from rs_tools._src.geoprocessing.modis.reproject import add_modis_crs
from rs_tools._src.utils.io import get_list_filenames

dask.config.set(**{"array.slicing.split_large_chunks": False})
warnings.filterwarnings("ignore", category=FutureWarning)


from rs_tools._src.data.modis import MODISFileName


@dataclass
class MODISGeoProcessorRadiances:
    file: MODISFileName

    @classmethod
    def init_from_file(cls, file: str):
        file = MODISFileName.from_filename(file)
        return cls(file=file)

    @property
    def radiances_raw(self) -> xr.Dataset:
        return load_modis_radiances_raw(self.file.full_path)

    @property
    def radiances_geoprocessed(
        self,
    ) -> xr.Dataset:
        return preprocess_modis_radiances(self.radiances_raw)


@dataclass
class MODISGeoProcessorClouds:
    file: MODISFileName

    @classmethod
    def init_from_file(cls, file: str):
        file = MODISFileName.from_filename(file)
        return cls(file=file)

    @property
    def cloud_mask_raw(self) -> xr.Dataset:
        return load_modis_cloud_mask_raw(self.file.full_path)


@dataclass
class MODISGeoProcessorPaired:
    radiance_geoprocessor: MODISGeoProcessorRadiances
    clouds_geoprocessor: MODISGeoProcessorClouds

    @classmethod
    def init_from_file(cls, radiances_file: str, clouds_file: str):
        radiance_geoprocessor = MODISGeoProcessorRadiances.init_from_file(
            radiances_file
        )
        clouds_geoprocessor = MODISGeoProcessorClouds.init_from_file(clouds_file)
        return cls(
            radiance_geoprocessor=radiance_geoprocessor,
            clouds_geoprocessor=clouds_geoprocessor,
        )

    @property
    def radiances_and_cloudmasks(self) -> xr.Dataset:
        # load radiances geoprocessed
        ds_rads = self.radiance_geoprocessor.radiances_geoprocessed

        ds_clouds = self.clouds_geoprocessor.cloud_mask_raw

        return combine_rads_and_clouds(ds_rads, ds_clouds)


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


def regrid_swath_to_regular(
    modis_swath: xr.Dataset,
    calibration: str="radiance",
    resolution: float = 0.01,
    radius_of_influence: int = 2_000,
    neighbours: int = 1,
    resample_type: str = "nn",
):

    # get footprint polygon
    swath_polygon = footprint(modis_swath.longitude.values, modis_swath.latitude.values)
    # create geometry
    odc_geom = Geometry(geom=swath_polygon, crs=CRS("4326"))

    gbox = GeoBox.from_geopolygon(odc_geom, resolution=resolution, crs=CRS("4326"))

    # create xarray coordinates
    coords = xr_coords(gbox)

    # create 2D meshgrid of coordinates
    LON, LAT = np.meshgrid(
        coords["longitude"].values, coords["latitude"].values, indexing="xy"
    )

    # create interpolation grids
    grid_def = GridDefinition(lons=LON, lats=LAT)
    swath_def = SwathDefinition(
        lons=modis_swath.longitude.values,
        lats=modis_swath.latitude.values,
        crs=modis_swath.rio.crs,
    )

    valid_input_index, valid_output_index, index_array, distance_array = (
        kd_tree.get_neighbour_info(swath_def, grid_def, radius_of_influence, neighbours)
    )

    def apply_resample(data):
        result = kd_tree.get_sample_from_neighbour_info(
            resample_type,
            grid_def.shape,
            data,
            valid_input_index,
            valid_output_index,
            index_array,
        )
        return result

    out = xr.apply_ufunc(
        apply_resample,
        modis_swath[calibration],
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y_new", "x_new"]],
        vectorize=True,
    )
    modis_grid = out.rename({"x_new": "x", "y_new": "y"}).to_dataset()
    modis_grid = modis_grid.assign_coords({"band_wavelength": modis_swath.band_wavelength})
    modis_grid = modis_grid.assign_coords({"time": modis_swath.time})

    modis_grid = modis_grid.rename({"x": "longitude", "y": "latitude"})
    modis_grid = modis_grid.assign_coords(
        {"longitude": coords["longitude"], "latitude": coords["latitude"]}
    )
    modis_grid = modis_grid.rio.write_crs(4326, inplace=True)

    return modis_grid


def regrid_to_regular(
    modis_swath: xr.DataArray,
    resolution: float = 0.01,
    radius_of_influence: int = 2_000,
    neighbours: int = 1,
    resample_type: str = "nn",
) -> xr.DataArray:
    


    # get footprint polygon
    swath_polygon = footprint(modis_swath.longitude.values, modis_swath.latitude.values)
    # create geometry
    odc_geom = Geometry(geom=swath_polygon, crs=CRS("4326"))

    gbox = GeoBox.from_geopolygon(odc_geom, resolution=resolution, crs=CRS("4326"))

    # create xarray coordinates
    coords = xr_coords(gbox)

    # create 2D meshgrid of coordinates
    LON, LAT = np.meshgrid(
        coords["longitude"].values, coords["latitude"].values, indexing="xy"
    )

    # create interpolation grids
    grid_def = GridDefinition(lons=LON, lats=LAT)
    swath_def = SwathDefinition(
        lons=modis_swath.longitude.values,
        lats=modis_swath.latitude.values,
        crs=modis_swath.rio.crs,
    )

    valid_input_index, valid_output_index, index_array, distance_array = (
        kd_tree.get_neighbour_info(swath_def, grid_def, radius_of_influence, neighbours)
    )

    def apply_resample(data):
        result = kd_tree.get_sample_from_neighbour_info(
            resample_type,
            grid_def.shape,
            data,
            valid_input_index,
            valid_output_index,
            index_array,
        )
        return result

    out = xr.apply_ufunc(
        apply_resample,
        modis_swath,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y_new", "x_new"]],
        vectorize=True,
    )
    modis_grid = out.rename({"x_new": "x", "y_new": "y"}).to_dataset()


    modis_grid = modis_grid.rename({"x": "longitude", "y": "latitude"})
    modis_grid = modis_grid.assign_coords(
        {"longitude": coords["longitude"], "latitude": coords["latitude"]}
    )
    modis_grid = modis_grid.rio.write_crs(4326, inplace=True)

    return modis_grid


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

    for var in ds_rads.data_vars:
        ds_rads[var].attrs = {}
    return ds_rads


def geoprocess_modis_aqua_terra(
        satellite_name: str, 
        calibration: str, 
        read_path: str, 
        save_path: str,
        ):
    """
    Geoprocesses MODIS Aqua/Terra data.

    Args:
        satellite_name (str): Name of the satellite ('Aqua' or 'Terra').
        calibration (str): Calibration type.
        read_path (str): Path to the MODIS RAW files.
        save_path (str): Path to save the processed data.

    Returns:
        None
    """
    # Initialize MODIS GeoProcessor
    logger.info(f"Initializing {satellite_name.upper()} GeoProcessor...")
    logger.debug(f"Path: {read_path}")
    # load the MODIS RAW Files Reader
    modis_files = MODISRawFiles(read_path)
    # grab all files
    modis_list_of_files = modis_files.modis_file_obj
    # filter for only satellites
    modis_list_of_files = list(
        filter(lambda x: MODIS_NAME_TO_ID[satellite_name] in x.satellite_id, modis_list_of_files)
    )

    logger.debug(f"Number of Files: {len(modis_list_of_files)}")

    pbar = tqdm(modis_list_of_files)

    for ifile in pbar:
        pbar.set_description(f"Satellite - {satellite_name} | Starting time: {ifile.datetime_acquisition}...")
        # skip if the satellite isn't correct
        if satellite_name != ifile.satellite_name:
            continue

        # load modis swath
        pbar.set_description(f"Loading MODIS SWATH...")
        modis_swath = load_modis_data_raw(ifile.full_path, calibration=calibration)

        # preprocess modis swath
        pbar.set_description(f"Cleaning Data...")
        modis_swath = preprocess_modis_raw(modis_swath, calibration=calibration).compute()

        # check if save path exists, and create if not
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # remove file if it already exists
        itime_name = datetime.strftime(ifile.datetime_acquisition, "%Y%m%d%H%M%S")
        save_filename = Path(save_path).joinpath(f"{itime_name}_{satellite_name}_{calibration}.nc")
        if os.path.exists(save_filename):
            logger.info(f"File already exists. Overwriting file: {save_filename}")
            os.remove(save_filename)

        modis_swath.to_netcdf(save_filename, engine="netcdf4")


def geoprocess(
    satellite_name: str = "aqua",
    calibration: str="radiance",
    read_path: str = "./",
    save_path: str = "./",
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
        geoprocess_modis_aqua_terra(satellite_name, calibration=calibration, read_path=read_path, save_path=save_path)
    elif satellite_name in ["aqua_cloud", "terra_cloud"]:
        raise NotImplementedError()
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
    typer.run(geoprocess)
