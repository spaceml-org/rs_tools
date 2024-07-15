from typing import Optional

import rioxarray
import xarray as xr
import numpy as np
from pyproj import CRS
from pyresample import kd_tree
from pyresample.geometry import (
    GridDefinition,
    SwathDefinition,
)
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox
from odc.geo.geom import (
    BoundingBox,
    Geometry,
)
from odc.geo.xr import xr_coords
from georeader.griddata import footprint
from dataclasses import dataclass


def add_modis_crs(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds the Coordinate Reference System (CRS) to the given MODIS dataset.

    Parameters:
    - ds (xarray.Dataset): The dataset to which the CRS will be added.

    Returns:
    - xarray.Dataset: The dataset with the CRS added.
    """
    # define CRS of MODIS dataset
    crs = 'WGS84'

    # load source CRS from the WKT string
    cc = CRS(crs)

    # assign CRS to dataarray
    ds.rio.write_crs(cc, inplace=True)

    return ds



def regrid_swath_to_regular(
    modis_swath: xr.Dataset,
    variable: str,
    resolution: float = 0.01,
    radius_of_influence: int = 2_000,
    neighbours: int = 1,
    resample_type: str = "nn",
    fill_value: float=0.0
):
    variable_attrs = modis_swath[variable].attrs

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
            resample_type=resample_type,
            output_shape=grid_def.shape,
            data=data,
            valid_input_index=valid_input_index,
            valid_output_index=valid_output_index,
            index_array=index_array,
            fill_value=fill_value
        )
        return result

    out = xr.apply_ufunc(
        apply_resample,
        modis_swath[variable],
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y_new", "x_new"]],
        vectorize=True,
    )
    modis_grid = out.rename({"x_new": "x", "y_new": "y"}).to_dataset()
    modis_grid[variable].attrs = variable_attrs
    # modis_grid = modis_grid.assign_coords({"band_wavelength": modis_swath.band_wavelength})
    modis_grid = modis_grid.assign_coords({"time": modis_swath.time})

    modis_grid = modis_grid.rename({"x": "longitude", "y": "latitude"})
    modis_grid = modis_grid.assign_coords(
        {"longitude": coords["longitude"], "latitude": coords["latitude"]}
    )
    modis_grid = modis_grid.rio.write_crs(4326, inplace=True)

    return modis_grid

