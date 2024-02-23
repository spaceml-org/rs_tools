import numpy as np

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple

from rs_tools import goes_download, modis_download, MODIS_VARIABLES

from rs_tools._src.geoprocessing.grid import create_latlon_grid

@dataclass
class GeoProcessingParams:
    # region of interest
    # target grid
    # unit conversion
    # crs transformation
    # resampling Transform
    region: Tuple[float, float, float, float] = ...
    save_path: str = 'path/to/bucket' # analysis ready bucket



@dataclass
class MODISGeoProcessing:
    resolution: float = ... # in degrees?
    save_path: str = ...

    def geoprocess(self, params: GeoProcessingParams, files: List[str]):
        save_path: str = params.save_path.join(self.save_path)
        target_grid: np.ndarray = create_latlon_grid(params.region, self.resolution) # create using np.linspace?

        # loop through files
        # open dataset
        # stack variables to channels
        # resample
        # convert units (before or after resampling???)
        # save as netcdf





@dataclass
class GOESGeoProcessing:
    resolution: float = ... # in degrees?

    def geoprocess(self, params: GeoProcessingParams, files: List[str]):
        return None
        
    def parse_filenames(self, files: List[str]):
        # chunk the files
        # time, bands
        return None

@dataclass
class MLProcessingParams:
    # patching
    # normalization
    # gap-filling
    pass

