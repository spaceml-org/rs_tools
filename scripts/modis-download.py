"""
Anna Tips 4 MODIS
- all bands are in a single file
- Format: hdf5 file
- resolution dependent [0.5km, 0.25km, 1km]
- downloader has the correct Level!
- time will be painful
   * multiple files 4 multiple SWATHS
   * very little revisit time during the day
- Filtering Locations / Bounding Boxes / Tiles
- Day & Night Flag: file sizes, filler value
- Level 1B - SWATH Product



potentially useful packages:
- modis-tools: https://github.com/fraymio/modis-tools
- pyMODIS: http://www.pymodis.org and https://github.com/lucadelu/pyModis



-------------------------------------------------------------------------
Summary: The download script to interact directly with the goes2go package.
We only want to specify what is necessary and compatible with the goes2go package.
In general, we want to satellite, the spatial domain, and the period.

Args:
    satellite_number: 

**Input Parameters**

Downloading:
- satellite number: int --> (16,17,18)
- spatial extent: str --> full disk (F), CONUS (C), Mesoscale domains (M, M1, M2)
- goes instrument: str --> e.g. ABI radiance (or SUVI for helio)
- preprocessing level: str --> e.g. level-1b
- directory: str
- return xarray dataset or list of files --> return as file list works better?
- band specifications: list[int] --> download all or subset only
- start time (of range of times to be downloaded): str
- end time: str
- timesteps/number of files: str
- day vs. night mode: --> e.g. for only downloading day mode images

----------

---
Basic Processing:
- resolution: --> downscale all bands to common resolution (e.g. 2 km)
- coordinate system transformations
- etc.


=================
INPUT PARAMETERS
=================

# LIST OF DATES | START DATE, END DATE, STEP
np.arange, np.linspace
t0, t1, dt | num_files
timestamps = [t0, t1, t2]
# create list of dates
list_of_dates: list = ["2020-10-19 12:00:00", "2020-10-12 12:00:00", ...]

# SATELLITE INFORMATION
satellite_number: int = (16, 17, 18)i
instrument: str  = (ABI, ...)
processing_level: str = (level-1b,): str = (level-1b, ...)
data_product: str = (radiances, ...)

# LIST OF BANDS
list_of_bands: list = [1, 2, ..., 15, 16]

# TARGET-GRID
target_grid: xr.Dataset = ...

% ===============
HOW DO WE CHECK DAYTIME HOURS?
* Get Centroid for SATELLITE FOV - FIXED
* Get Radius points for SATELLITE FOV - FIXED
* Check if centroid and/or radius points for FOV is within daytime hours
* Add warning if chosen date is before GOES orbit was stabilized
* True:
  download that date
False:
    Skippppp / Continue
    
@dataclass
class SatelliteFOV:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    viewing_angle: ... # [0.15, -0.15] [rad]

    @property
    def get_radius(self):
        ...

class GOES16(SatelliteFOV):
    ...

=================
ALGORITHMS
=================

for itime in timestamps:
    for iband in list_of_bands:
        # -------------------------------------------------
        # download to folder - use GOES2GO loader
        # -------------------------------------------------
        
        # open data in folder
        
        # -------------------------------------------------
        # quality check 1 - did it download
        # -------------------------------------------------
        if download_criteria:
            continue if allow_missing else break
        
        # quality check 2 - day and/or night specification
        if day_night_criteria:
            continue if allow_missing else break
        
        # -------------------------------------------------
        # CRS Transformation (Optional, preferred)
        # -------------------------------------------------
        # load dataset
        ds: xr.Dataset = xr.load_dataset(...)
        # coordinate transformation
        ds: xr.Dataset = crs_transform(ds, target_crs, *args, **kwargs)
        # resave
        ds.to_netcdf(...)
        
        # -------------------------------------------------
        # downsample/upscale/lower-res (optional, preferred)
        # -------------------------------------------------
        # load dataset
        ds: xr.Dataset = xr.load_dataset(...)
        # resample
        ds: xr.Dataset = downsample(ds, target_grid, *args, **kwargs)
        ds: xr.Dataset = transform_coords(ds, target_coords)
        # resave
        ds.to_netcdf(...)
"""

from typing import Optional, List, Union
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import typer
from loguru import logger
from datetime import datetime, timedelta

## SAT02XXX.AYYYYDD.HHDD.061.??????????????????.hdf
# MOD - TERRA
# MYD - AQUA
# XXX = 1KM (QKM, HKM)
# server - https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/
MOD021KM/YEAR/DAY

def modis_download(
    start_date: str,
    end_date: Optional[str]=None,
    start_time: Optional[str]='00:00:00',
    end_time: Optional[str]='23:59:00',
    day_step: Optional[str]=None,
    satellite: str='Terra',
    save_dir: Optional[str]=".",
    processing_level: str = 'L1b',
    resolution: str = "1KM",
    collection: str = '61'
):
    # run checks
    _check_input_processing_level(processing_level=processing_level)
    satellite_code = _check_satellite(satellite=satellite)
    resolution_code = _check_resolution(resolution=resolution)
    logger.info(f"Satellite: {satellite}")
    # check data product
    data_product = f"{satellite_code}02{resolution_code}"
    logger.info(f"Data Product: {data_product}")
    _check_data_product_name(data_product=data_product)

    # check start/end dates/times
    if end_date is None:
        end_date = start_date

    _check_date_format(start_date, end_date)

    # datetime conversion 
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    _check_start_end_dates(start_datetime=start_datetime, end_datetime=end_datetime)
          

    def _check_input_processing_level(processing_level: str) -> bool:
        """checks processing level for MODIS data"""
        if processing_level in ["L1b"]:
            return True
        else:
            msg = "Unrecognized processing level"
            msg += f"\nNeeds to be 'L1b'. Others are not yet implemented"
            raise ValueError(msg)

    def _check_satellite(satellite: str) -> str:
        if satellite == 'Aqua': 
            return 'MYD'
        elif satellite == 'Terra':
            return 'MOD'
        else:
            msg = "Unrecognized satellite"
            msg += f"\nNeeds to be 'Aqua' or 'Terra'. Others are not yet implemented"
            raise ValueError(msg)
        
    def _check_resolution(resolution: str) -> str:
        if resolution in ["1KM", "1Km", "1km"]:
            return "1KM"
        elif resolution in ["500M", "500m"]:
            return "HKM"
        elif resolution in ["250M", "250m"]:
            return "QKM"
        else: 
            msg = "Unrecognized resolution"
            msg += f"\nNeeds to be '1KM', '500M', '250M. Others are not available"
            raise ValueError(msg)
        
    def _check_data_product_name(data_product: str) -> bool:
        if data_product in ['MOD021KM', 'MOD02HKM', 'MOD02QKM', 'MYD021KM', 'MYD02HKM', 'MYD02QKM']:
            return True
        else:
            msg = "Unrecognized data product"
            msg += f"\nOnly implemented for TERRA/AQUA MODIS and 1KM, 500M, 250M resolution."
            raise ValueError(msg)

    def _check_date_format(start_date: str, end_date: str) -> bool:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
            return True
        except Exception as e:
            msg = "Please check date/time format"
            msg += "\nExpected date format: %Y-%m-%d"
            raise SyntaxError(msg)

    def _check_start_end_dates(start_datetime: datetime, end_datetime: datetime) -> bool:
        """ check end_datetime is after start_datetime """
        if start_datetime < end_datetime:
            return True
        else:
            msg = "Start datetime must be before end datetime\n"
            msg += f"This does not hold for start = {str(start_datetime)} and end = {str(end_datetime)}"
            raise ValueError(msg)
    

