import autoroot
import os
import earthaccess
from goes2go import GOES # activate the rio accessor
import rioxarray  # activate the rio accessor
import xarray as xr
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
import tqdm
import geopandas as gpd
from pathlib import Path
from loguru import logger


# PLOTTING
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import rasterio
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

# GOES2GO
import goes2go
from rs_tools._src.data.goes.download import goes_download

# GEOREADER
from georeader.dataarray import fromDataArray, toDataArray
from georeader.read import read_reproject, read_reproject_like
from georeader.griddata import read_to_crs, footprint

# RESAMPLE
from pyresample import kd_tree
from pyresample.geometry import SwathDefinition, GridDefinition
import numpy as np
import typer
# GEOBOX
from odc.geo.xr import xr_coords
from odc.geo.geom import BoundingBox
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.crs import CRS

from odc.geo.data import ocean_geom, country_geom

from rs_tools._src.geoprocessing.goes.reproject import add_goes16_crs
from rs_tools._src.geoprocessing.goes.validation import correct_goes16_bands, correct_goes16_satheight


def clean_coords_goes16(ds):
    # convert measurement angles to horizontal distance in meters
    ds = correct_goes16_satheight(ds) 
    try:
        # correct band coordinates to reorganize xarray dataset
        ds = correct_goes16_bands(ds) 
    except AttributeError:
        pass
    # assign coordinate reference system
    ds = add_goes16_crs(ds)

    return ds


def preprocess_goes16_radiances(ds):
    from pyproj import CRS
    # select radiances
    variables = ["Rad"] # "Rad" = radiance, "DQF" = data quality flag

    # Extract relevant attributes from original dataset
    time_stamp = pd.to_datetime(ds.t.values) 
    band_attributes = ds.band.attrs
    band_wavelength_attributes = ds.band_wavelength.attrs
    band_wavelength_values = ds.band_wavelength.values
    band_values = ds.band_id.values
    ds_fov = ds.FOV

    cc = CRS.from_cf(ds.goes_imager_projection.attrs)

    # do core preprocess function (e.g. to correct band coordinates)
    ds = clean_coords_goes16(ds)

    # convert measurement time (in seconds) to datetime
    time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M") 
    # assign bands data to each variable
    ds = ds[variables]
    ds = ds.expand_dims({"band": band_values})
    # attach time coordinate
    ds = ds.assign_coords({"time": [time_stamp]})
    # drop variables that will no longer be needed
    # load CRS
    # assign CRS to dataarray

    
    ds = ds.drop_vars(["t", "y_image", "x_image", "goes_imager_projection", "time_coverage_start", "time_coverage_end", "date_created", "dataset_name"])
    # assign band attributes to dataset
    ds.band.attrs = band_attributes
    # assign band wavelength to each variable
    ds = ds.assign_coords({"band_wavelength": band_wavelength_values})
    ds.band_wavelength.attrs = band_wavelength_attributes
    
    # write crs
    ds.rio.write_crs(cc.to_string(), inplace=True)

    # Keep only certain relevant attributes
    attrs_rad = ds["Rad"].attrs

    ds["Rad"].attrs = {}
    ds["Rad"].attrs = dict(
        long_name=attrs_rad["long_name"],
        standard_name=attrs_rad["standard_name"],
        units=attrs_rad["units"],
    )
    ds.attrs = {}
    return ds


def resample_modis_swath_to_grid(
        modis_swath: xr.Dataset, 
        resolution: float=0.01,
        radius_of_influence: int=2000,
        resample_type: str="nn"
        ):

    # get footprint polygon
    swath_polygon = footprint(modis_swath.longitude.values, modis_swath.latitude.values)
    
    # create geometry
    odc_geom = Geometry(geom=swath_polygon, crs=CRS("4326"))

    # initialize geobox
    gbox = GeoBox.from_geopolygon(odc_geom, resolution=resolution, crs=CRS("4326"))

    # create xarray coordinates
    coords = xr_coords(gbox)

    # create 2D meshgrid of coordinates
    LON, LAT  = np.meshgrid(coords["longitude"].values, coords["latitude"].values, indexing="xy")

    # create interpolation grids
    grid_def = GridDefinition(lons=LON, lats=LAT)
    swath_def = SwathDefinition(lons=modis_swath.longitude.values, lats=modis_swath.latitude.values)

    valid_input_index, valid_output_index, index_array, distance_array = \
        kd_tree.get_neighbour_info(swath_def, grid_def, radius_of_influence, neighbours=1)
    

    def apply_resample(data): 
        result = kd_tree.get_sample_from_neighbour_info(
            resample_type, 
            grid_def.shape, 
            data, 
            valid_input_index, 
            valid_output_index, 
            index_array
        )
        return result

    out = xr.apply_ufunc(
        apply_resample, 
        modis_swath.Rad,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y_new", "x_new"]],
        vectorize=True,
    )
    modis_grid = out.rename({"x_new": "x", "y_new": "y"}).to_dataset()

    modis_grid["band_wavelength"] = modis_swath.band_wavelength #
    modis_grid["time"] = modis_swath.time
    modis_grid = modis_grid.rename({"x": "longitude", "y": "latitude"})
    modis_grid = modis_grid.assign_coords({"longitude": coords["longitude"], "latitude": coords["latitude"]})
    modis_grid = modis_grid.rio.write_crs(4326, inplace=True)

    return modis_grid


def get_domain_intersection(modis_image, goes_image):

    # create MODIS GeoTensor
    modis_geotensor = fromDataArray(
        modis_image.Rad,
        crs=modis_image.rio.crs,
        x_axis_name="longitude",
        y_axis_name="latitude",
        fill_value_default=-999
    )

    
    # create geotensor
    goes_geotensor = fromDataArray(
        x=goes_image.Rad,
        crs=goes_image.rio.crs,
        fill_value_default=-999,
        x_axis_name="x", 
        y_axis_name="y"
    )

    # create a polygon for modis image
    modis_gdf = gpd.GeoDataFrame(index=[0], crs=modis_geotensor.crs, geometry=[modis_geotensor.footprint()])

    # create a polygon for GOES image
    goes_gdf = gpd.GeoDataFrame(index=[0], crs=goes_geotensor.crs, geometry=[goes_geotensor.footprint()])

    # project this polygon onto the GOES CRS
    modis_on_goes_gdf = modis_gdf.to_crs(goes_geotensor.crs)

    # find the intersection of this reprojected MODIS polygon and GOES polygon
    modis_on_goes_gdf = modis_on_goes_gdf.intersection(goes_gdf.unary_union)

    # subset GOES image based on MODIS subsection
    goes_image = goes_image.Rad.rio.clip(modis_on_goes_gdf.values, modis_on_goes_gdf.crs, drop=True, invert=False)

    return goes_image


def main(
        modis_file: str,
        resample_type: str="nn",
        radius_of_influence: int=2_000,
        resolution: float=0.01,
        save_path: str="./"
        ):

    # load modis image
    modis_file = Path(modis_file)
    logger.debug(f"Loading: {modis_file}")
    logger.info(f"Loading MODIS SWATH...")
    modis_swath = xr.open_dataset(filename_or_obj=modis_file, engine="netcdf4")

    # resample modis swath to grid
    logger.info(f"Resample MODIS SWATH to GRID...")
    modis_grid = resample_modis_swath_to_grid(
        modis_swath, 
        resolution=resolution, 
        radius_of_influence=radius_of_influence, 
        resample_type=resample_type
        )

    # grab query time
    attime = str(modis_grid.time.values.squeeze())
    logger.info(f"Querying time: {attime}...")


    pbar = tqdm.tqdm(list(enumerate(list(np.arange(1, 17)))))

    for i, iband in pbar:
        
        # load GOES Band
        G = GOES(
            satellite=16, 
            product="ABI-L1b-Rad", 
            domain='F',
            channel=iband
        )

        within = pd.to_timedelta(10, unit="minute")

        goes_imag_i = G.nearesttime(
            attime=attime,
            within=within,
        )

        # preprocess goes image
        goes_imag_i = preprocess_goes16_radiances(goes_imag_i)

        # get subset
        goes_imag_i = get_domain_intersection(modis_grid, goes_imag_i)

        if i == 0:
            goes_imag = goes_imag_i
        else:
            goes_imag_i = goes_imag_i.interp(x=goes_imag.x, y=goes_imag.y)
            goes_imag = xr.concat([goes_imag, goes_imag_i], dim="band")
            del goes_imag_i

    # goes_images = xr.combine_by_coords(goes_images)
    save_path = Path(save_path).joinpath("goes_sub_temp.nc")
    logger.debug(f"Saving data: {save_path}")
    goes_imag.to_netcdf(save_path, engine="netcdf4")


if __name__ == '__main__':
    typer.run(main)
