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

Questions:
- Are night/day mode measurments consistent?
    - no. They change slightly each day.
- Are the location measurements at specific times consistent?
    - no.



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

TODO - add changes to input parameters etcetera
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
import earthaccess

## SAT02XXX.AYYYYDD.HHDD.061.??????????????????.hdf
# MOD - TERRA
# MYD - AQUA
# XXX = 1KM (QKM, HKM)
# server - https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/

def modis_download(
    start_date: str,
    end_date: Optional[str]=None,
    start_time: Optional[str]='00:00:00', # used for daily window
    end_time: Optional[str]='23:59:00', # used for daily window
    day_step: Optional[int]=1, 
    satellite: str='Terra',
    save_dir: Optional[str]=".",
    processing_level: str = 'L1b',
    resolution: str = "1KM",
    earthdata_username: Optional[str]='',
    earthdata_password: Optional[str]='',
    bounding_box: Optional[tuple[float, float, float, float]]=(-180, -90, 180, 90), # TODO: extend to allow multiple regions? NOTE: earthaccess allows other ways to specify spatial extent, e.g. polygon, point - consider extending to allow these options
    # day_night_flag: Optional[str]=None, NOTE: can pass day/night flag  ('day' or 'night') but if arg is passed it cannot be None - need to find a way to make it work as optional argument
):
    # check if earthdata login is available
    _check_earthdata_login(earthdata_username=earthdata_username, earthdata_password=earthdata_password)

    # check if netcdf4 backend is available
    _check_netcdf4_backend()

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

    start_datetime_str = start_date + ' ' + start_time
    end_datetime_str = end_date + ' ' + end_time
    _check_datetime_format(start_datetime_str=start_datetime_str, end_datetime_str=end_datetime_str) 

    # datetime conversion
    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M:%S")

    _check_start_end_dates(start_datetime=start_datetime, end_datetime=end_datetime)
    
    # compile list of dates/times 
    day_delta = timedelta(days=day_step)
    list_of_dates = np.arange(start_datetime, end_datetime, day_delta).astype(datetime)
    
    # start/end times are used as daily window
    def get_daily_window(daily_start, end_time):
        """computes tuple of start and end date/time for each day for earthaccess call"""
        day = daily_start.strftime("%Y-%m-%d")
        daily_end = day + ' ' + end_time
        return (daily_start.strftime("%Y-%m-%d %H:%M:%S"), daily_end)
        
    list_of_daily_windows = [get_daily_window(daily_start, end_time) for daily_start in list_of_dates]


    # check if save_dir is valid before attempting to download
    # TODO could also check if save_dir exists and otherwise create it (assuming its parent directory exists, otherwise throw error)
    _check_save_dir(save_dir=save_dir)

    # check that bounding box is valid
    _check_bounding_box(bounding_box=bounding_box)
   
    files = []

    # create progress bar for dates
    pbar_time = tqdm.tqdm(list_of_daily_windows)


    for itime in pbar_time:
        pbar_time.set_description(f"Time - {itime[0]} to {itime[1]}")
        success_flag = True

        results_day = earthaccess.search_data(
            short_name=data_product,
            bounding_box=bounding_box,
            temporal=itime,
        )

        if not results_day:
            # check if any results were returned, if not: log warning and continue to next date
            success_flag = False
            logger.warning(f"No data found for {itime[0]} to {itime[1]} in the specified bounding box")
            continue

        files_day = earthaccess.download(results_day, save_dir) # TODO: can this fail? if yes, use try / except to prevent the programme from crashing
        # TODO: check file sizes - if less than X MB (ca 70MB) the download failed
        # TODO: are there any other checks we need to do here?
        if success_flag:
            files += files_day
    
    return files       



def _check_earthdata_login(earthdata_username: str, earthdata_password: str) -> bool:
    """check if earthdata login is available in environment variables / as input arguments"""
    if earthdata_username and earthdata_password:
        os.environ["EARTHDATA_USERNAME"] = earthdata_username
        os.environ["EARTHDATA_PASSWORD"] = earthdata_password
    
    if os.environ.get("EARTHDATA_USERNAME") is None or os.environ.get("EARTHDATA_PASSWORD") is None:
        msg = "Please set your Earthdata credentials as environment variables using:"
        msg += "\nexport EARTHDATA_USERNAME=<your username>"
        msg += "\nexport EARTHDATA_PASSWORD=<your password>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--earthdata-username <your username> --earthdata-password <your password>"
        raise ValueError(msg)
    
    # check if credentials are valid
    auth_obj = earthaccess.login('environment')

    if auth_obj.authenticated: 
        return True
    else:
        msg = "Earthdata login failed."
        msg += "\nPlease check your credentials and set them as environment variables using:"
        msg += "\nexport EARTHDATA_USERNAME=<your username>"
        msg += "\nexport EARTHDATA_PASSWORD=<your password>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--earthdata-username <your username> --earthdata-password <your password>"
        raise ValueError(msg)

def _check_netcdf4_backend() -> bool:
    """check if xarray netcdf4 backend is available"""
    if 'netcdf4' in xr.backends.list_engines().keys():
        return True
    else:
        msg = "Please install netcdf4 backend for xarray using one of the following commands:"
        msg += "\npip install netCDF4"
        msg += "\nconda install -c conda-forge netCDF4"
        raise ValueError(msg)

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

def _check_datetime_format(start_datetime_str: str, end_datetime_str: str) -> bool:
    try:
        datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
        datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M:%S")
        return True
    except Exception as e:
        msg = "Please check date/time format"
        msg += "\nExpected date format: %Y-%m-%d"
        msg += "\nExpected time format: %H:%M:%S"
        raise SyntaxError(msg)

def _check_start_end_dates(start_datetime: datetime, end_datetime: datetime) -> bool:
    """ check end_datetime is after start_datetime """
    if start_datetime < end_datetime:
        return True
    else:
        msg = "Start datetime must be before end datetime\n"
        msg += f"This does not hold for start = {str(start_datetime)} and end = {str(end_datetime)}"
        raise ValueError(msg)

def _check_bounding_box(bounding_box: List[float]) -> bool:
    """ check if bounding box is valid """
    lower_left_lon, lower_left_lat , upper_right_lon, upper_right_lat = bounding_box
    
    # check that latitudes and longitudes are within valid range
    if lower_left_lon < -180 or upper_right_lon > 180 or lower_left_lat < -90 or upper_right_lat > 90:
        msg = "Bounding box must be between -180 and 180 for longitude and -90 and 90 for latitude"
        msg += f"\nReceived: [lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat] = {bounding_box} "
        raise ValueError(msg)
    
    # check that upper lat is above lower lat
    if lower_left_lat > upper_right_lat:
        msg = "The bounding box north value ({upper_right_lat}) must be greater than the south value ({lower_left_lat})"
        msg = "Bounding box must be in the format [lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat]"
        msg += f"\nReceived: {bounding_box}"
        raise ValueError(msg)
    
    # warn if bounding box crosses the dateline
    if lower_left_lon > upper_right_lon:
        logger.warning(f"The bounding box crosses the dateline: it ranges from {lower_left_lon} to {upper_right_lon} degrees longitude")

    return True

    
    

def _check_save_dir(save_dir: str) -> bool:
    """ check if save_dir exists """
    if os.path.isdir(save_dir):
        return True
    else:
        msg = "Save directory does not exist"
        msg += f"\nReceived: {save_dir}"
        raise ValueError(msg)
    


if __name__ == '__main__':
    typer.run(modis_download)

    """
    # one day - successfully downloaded 4 granules (all nighttime)
    python scripts/modis-download.py 2018-10-01 --start-time 08:00:00 --end-time 8:10:00 --save-dir ./notebooks/modisdata/test_script/

    # multiple days - finds 62 granules, stopped download for times sake but seemed to work
    python scripts/modis-download.py 2018-10-01 --end-date 2018-10-9 --day-step 3 --start-time 08:00:00 --end-time 13:00:00 --save-dir ./notebooks/modisdata/test_script/

    # test bounding box - successfully downloaded 4 files (all daytime)
    python scripts/modis-download.py 2018-10-01 --start-time 08:00:00 --end-time 13:00:00 --save-dir ./notebooks/modisdata/test_script/ --bounding-box -10 -10 20 5


    # ====================
    # FAILURE TEST CASES
    # ====================
    # bounding box input invalid - throws error as expected
    python scripts/modis-download.py 2018-10-01 --bounding-box a b c d

    # end date before start date - throws error as expected
    python scripts/modis-download.py 2018-10-01  --end-date 2018-09-01 

    # empty results - warns user as expected
    python scripts/modis-download.py 2018-10-01 --start-time 07:00:00 --end-time 7:10:00 --save-dir ./notebooks/modisdata/test_script/ --bounding-box -10 -10 -5 -5

    """
