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
from rs_tools.modis import get_modis_channel_numbers, CALIBRATION_CHANNELS, VARIABLE_ATTRS
from rs_tools._src.data.modis import (
    MODIS_ID_TO_NAME,
    MODIS_NAME_TO_ID,
    MODISFileName,
    MODISRawFiles,
    get_modis_paired_files,
)
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

