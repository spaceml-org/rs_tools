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
    bounding_box: Optional[tuple[float, float, float, float]]=(-180, -90, 180, 90), # TODO: Add polygon option
    earthdata_username: Optional[str]="",
    earthdata_password: Optional[str]="",
    day_night_flag: Optional[str]=None, 
):
    """
    Downloads MODIS satellite data for a specified time period and location.

    Args:        
        start_date (str): The start date of the data download in the format 'YYYY-MM-DD'.
        end_date (str, optional): The end date of the data download in the format 'YYYY-MM-DD'. If not provided, the end date will be the same as the start date.
        start_time (str, optional): The start time of the data download in the format 'HH:MM:SS'. Default is '00:00:00'.
        end_time (str, optional): The end time of the data download in the format 'HH:MM:SS'. Default is '23:59:00'.
        day_step (int, optional): The time step (in days) between downloads. This is to allow the user to download data every e.g. 2 days. If not provided, the default is daily downloads.
        satellite (str, optional): The satellite. Options are "Terra" and "Aqua", with "Terra" as default.
        save_dir (str, optional): The directory where the downloaded files will be saved. Default is the current directory.
        processing_level (str, optional): The processing level of the data. Default is 'L1b'.
        resolution (str, optional): The resolution of the data. Options are "QKM" (250m), "HKM (500m), "1KM" (1000m), with "1KM" as default. Not all bands are measured at all resolutions.
        bounding_box (tuple, optional): The region to be downloaded.
        earthdata_username (str): Username associated with the NASA Earth Data login. Required for download.
        earthdata_password (str): Password associated with the NASA Earth Data login. Required for download.
        
    Returns:
        list: A list of file paths for the downloaded files.
        
    Examples:
    # one day - successfully downloaded 4 granules (all nighttime)
    python scripts/modis-download.py 2018-10-01 --start-time 08:00:00 --end-time 8:10:00 --save-dir ./notebooks/modisdata/test_script/

    # multiple days - finds 62 granules, stopped download for times sake but seemed to work
    python scripts/modis-download.py 2018-10-01 --end-date 2018-10-9 --day-step 3 --start-time 08:00:00 --end-time 13:00:00 --save-dir ./notebooks/modisdata/test_script/

    # test bounding box - successfully downloaded 4 files (all daytime)
    python scripts/modis-download.py 2018-10-01 --start-time 08:00:00 --end-time 13:00:00 --save-dir ./notebooks/modisdata/test_script/ --bounding-box -10 -10 20 5

    # test day/night flag - successfully downloaded 1 file (daytime only)
    python scripts/modis-download.py 2018-10-15 --save-dir ./notebooks/modisdata/test_script/ --bounding-box -10 10 -5 15 --day-night-flag day


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
    # check if earthdata login is available
    _check_earthdata_login(earthdata_username=earthdata_username, earthdata_password=earthdata_password)

    # check if netcdf4 backend is available
    _check_netcdf4_backend()

    # run checks
    # translate str inputs to modis specific names
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

    # combine date and time information
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
    
    list_of_daily_windows = [get_daily_window(daily_start, end_time) for daily_start in list_of_dates]

    # check if save_dir is valid before attempting to download
    _check_save_dir(save_dir=save_dir)

    # check that bounding box is valid
    # TODO: Add option to add multiple location requests
    # NOTE: earthaccess allows other ways to specify spatial extent, e.g. polygon, point - consider extending to allow these options
    _check_bounding_box(bounding_box=bounding_box)

    # create dictionary of earthaccess search parameters
    search_params = {
        "short_name": data_product,
        "bounding_box": bounding_box,
    }

    # if day_night_flag was provided, check that day_night_flag is valid
    if day_night_flag: 
        _check_day_night_flag(day_night_flag=day_night_flag)
        # add day_night_flag to search parameters
        search_params["day_night_flag"] = day_night_flag

    # TODO remove - logging search_params for testing
    logger.info(f"Search parameters: {search_params}")
   
    files = []

    # create progress bar for dates
    pbar_time = tqdm.tqdm(list_of_daily_windows)

    for itime in pbar_time:
        pbar_time.set_description(f"Time - {itime[0]} to {itime[1]}")
        success_flag = True

        # add daytime window to search parameters
        search_params["temporal"] = itime

        # search for data
        results_day = earthaccess.search_data(**search_params)

        # check if any results were returned
        if not results_day:
            # if not: log warning and continue to next date
            success_flag = False
            warn = f"No data found for {itime[0]} to {itime[1]} in the specified bounding box"
            if day_night_flag: warn += f" for {day_night_flag}-time measurements only"
            logger.warning(warn)
            continue

        files_day = earthaccess.download(results_day, save_dir) 
        # TODO: can this fail? if yes, use try / except to prevent the programme from crashing
        # TODO: check file sizes - if less than X MB (ca 70MB) the download failed
        # TODO: Add check for day/night/mixed mode measurements
        if success_flag:
            files += files_day
    
    return files    

# start/end times are used as daily window
def get_daily_window(daily_start, end_time):
    """computes tuple of start and end date/time for each day for earthaccess call"""
    day = daily_start.strftime("%Y-%m-%d")
    daily_end = day + ' ' + end_time
    return (daily_start.strftime("%Y-%m-%d %H:%M:%S"), daily_end)
    

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
        try:
            os.mkdir(save_dir)
            return True
        except:
            msg = "Save directory does not exist"
            msg += f"\nReceived: {save_dir}"
            msg += "\nCould not create directory"
            raise ValueError(msg)
        
def _check_day_night_flag(day_night_flag: str) -> bool:
    """ check if day_night_flag is valid """
    if day_night_flag in ["day", "night"]:
        return True
    else:
        msg = "Unrecognized day/night flag"
        msg += f"\nReceived: {day_night_flag}"
        msg += f"\nIf provided, it needs to be 'day' or 'night'."
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
