from typing import Optional, List, Union
import os
import warnings
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import shutil
import typer
import eumdac
from loguru import logger
from datetime import datetime, timedelta

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


def msg_download(
    start_date: Optional[str]=None, # TODO: Wrap date/time parameters in dict to reduce arguments?
    end_date: Optional[str]=None,
    start_time: Optional[str]='00:00:00', # EUMDAC did not find any data for 00:00:00
    end_time: Optional[str]='23:59:00', # EUMDAC did not find any data for 23:59:00
    daily_window_t0: Optional[str]='00:00:00',
    daily_window_t1: Optional[str]='23:59:00',
    time_step: Optional[str]=None,
    predefined_timestamps: Optional[List]=None,
    satellite: str="MSG",
    instrument: str ="HRSEVIRI",
    processing_level: Optional[str] = "L1",
    save_dir: Optional[str] = ".",
    eumdac_key: Optional[str]="",
    eumdac_secret: Optional[str]="",
):
    """
    Downloads MSG satellite data for a specified time period and set of bands.

    Args:
        start_date (str, optional): The start date of the data download in the format 'YYYY-MM-DD'.
        end_date (str, optional): The end date of the data download in the format 'YYYY-MM-DD'. If not provided, the end date will be the same as the start date.
        start_time (str, optional): The start time of the data download in the format 'HH:MM:SS'. Default is '00:00:00'.
        end_time (str, optional): The end time of the data download in the format 'HH:MM:SS'. Default is '23:59:00'.
        daily_window_t0 (str, optional): The start time of the daily window in the format 'HH:MM:SS'. Default is '00:00:00'. Used if e.g., only day/night measurements are required.
        daily_window_t1 (str, optional): The end time of the daily window in the format 'HH:MM:SS'. Default is '23:59:00'. Used if e.g., only day/night measurements are required.
        time_step (str, optional): The time step between each data download in the format 'HH:MM:SS'. If not provided, the default is 1 hour.
        predefined_timestamps (list, optional): A list of timestamps to download. Expected format is datetime or str following 'YYYY-MM-DD HH:MM:SS'. If provided, start/end dates/times will be ignored.
        satellite (str): The satellite. Default is MSG.
        instrument (str): The data product to download. Default is 'HRSEVIRI', also implemented for Cloud Mask (CLM).
        processing_level (str, optional): The processing level of the data. Default is 'L1'.
        save_dir (str, optional): The directory where the downloaded files will be saved. Default is the current directory.
        eumdac_key (str, optional): The EUMETSAT Data Centre (EUMDAC) API key. If not provided, the user will be prompted to enter the key.
        eumdac_secret (str, optional): The EUMETSAT Data Centre (EUMDAC) API secret. If not provided, the user will be prompted to enter the secret.

    Returns:
        list: A list of file paths for the downloaded files.
        
    Examples:
        # =========================
        # MSG LEVEL 1 Test Cases
        # =========================
        # custom day
        python scripts/msg-download.py 2018-10-01
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01
        # custom day + end points
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 09:00:00 --end-time 12:00:00
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --start-time 09:05:00 --end-time 12:05:00
        # custom day + end points + time window
        scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 00:05:00 --end-time 23:54:00 --daily-window-t0 09:00:00 --daily-window-t1 12:00:00 
        # custom day + end points + time window + timestep
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 00:05:00 --end-time 23:54:00 --daily-window-t0 09:00:00 --daily-window-t1 12:00:00 --time-step 00:15:00
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 00:05:00 --end-time 23:54:00 --daily-window-t0 09:00:00 --daily-window-t1 12:00:00 --time-step 00:25:00
        # ===================================
        # MSG CLOUD MASK Test Cases
        # ===================================
        # custom day
        python scripts/msg-download.py 2018-10-01 --instrument=CLM
        # custom day + end points
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --instrument=CLM 
        # custom day + end points + time window
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --start-time 09:00:00 --end-time 12:00:00 --instrument=CLM 
        # custom day + end points + time window + timestep
        python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --start-time 09:00:00 --end-time 12:00:00 --time-step 00:25:00 --instrument=CLM
        # ====================
        # FAILURE TEST CASES
        # ====================
    """

    # run checks
    # check if eumdac login is available
    token = _check_eumdac_login(eumdac_key=eumdac_key, eumdac_secret=eumdac_secret)
    datastore = eumdac.DataStore(token)

    # check if netcdf4 backend is available
    _check_netcdf4_backend()

    # check satellite details
    _check_input_processing_level(processing_level=processing_level)
    _check_instrument(instrument=instrument)
    # check data product
    data_product = f"EO:EUM:DAT:{satellite}:{instrument}"
    logger.info(f"Data Product: {data_product}")
    _check_data_product_name(data_product=data_product)

    # TODO: Allow passing as argument?
    timestamp_dict = { 
        'start_date': start_date,
        'end_date': end_date,
        'start_time': start_time,
        'end_time': end_time,
        'daily_window_t0': daily_window_t0,
        'daily_window_t1': daily_window_t1,
        'time_step': time_step,
    }

    # compile list of dates
    list_of_dates = _compile_list_of_dates(timestamp_dict=timestamp_dict, predefined_timestamps=predefined_timestamps)

    # check if save_dir is valid before attempting to download
    _check_save_dir(save_dir=save_dir)

    successful_queries = []
    files = []

    # create progress bars for dates and bands
    pbar_time = tqdm.tqdm(list_of_dates)

    for itime in pbar_time:
        
        pbar_time.set_description(f"Time - {itime}")

        sub_files_list = _download(time=itime, data_product=data_product, save_dir=save_dir, datastore=datastore)
        if sub_files_list is None:
            logger.info(f"Could not find data for time {itime}. Trying to remove 5 mins from timestamp {itime}.")
            time_delta = timedelta(hours=0, minutes=5, seconds=0)
            itime_minus5 = itime - time_delta
            sub_files_list = _download(time=itime_minus5, data_product=data_product, save_dir=save_dir, datastore=datastore)
        if sub_files_list is None:
            logger.info(f"Could not find data for time {itime_minus5}. Trying to add 5 mins to timestamp {itime}.")
            time_delta = timedelta(hours=0, minutes=5, seconds=0)
            itime_plus5 = itime+ time_delta
            sub_files_list = _download(time=itime_plus5, data_product=data_product, save_dir=save_dir, datastore=datastore)

        if sub_files_list is None:
            logger.info(f"Could not find data for time {itime}. Skipping to next time.")
        else:
            files += sub_files_list
            successful_queries.append(itime)

    return (files, successful_queries)

def _download(time: datetime, data_product: str, save_dir: str, datastore):
    products = _compile_msg_products(data_product=data_product, time=time, datastore=datastore)
    sub_files_list = _msg_data_download(products=products, save_dir=save_dir)
    return sub_files_list

def _compile_msg_products(data_product: str, time: datetime, datastore):
    selected_collection = datastore.get_collection(data_product)
    products = selected_collection.search(
        dtstart=time,
        dtend=time)
    return products

def _msg_data_download(products, save_dir: str):
    try:
        for product in products:
            for entry in product.entries:
                if entry.endswith(".nat") or entry.endswith(".grb"): 
                    with product.open(entry=entry) as fsrc:
                        # Create a full file path for saving the file
                        save_path = os.path.join(save_dir, os.path.basename(fsrc.name))
                        # Check if file already exists
                        if os.path.exists(save_path):
                            print(f"File {save_path} already exists. Skipping download.")
                            return [save_path]
                        else:
                            with open(save_path, mode='wb') as fdst:
                                shutil.copyfileobj(fsrc, fdst)
                            print(f"Successfully downloaded {entry}.")
                            return [save_path]
    except Exception as error:
        print(f"Error downloading product': '{error}'")
        pass

def _check_eumdac_login(eumdac_key: str, eumdac_secret: str) -> bool:
    """check if eumdac login is available in environment variables / as input arguments"""
    if eumdac_key and eumdac_key:
        os.environ["EUMDAC_KEY"] = eumdac_key
        os.environ["EUMDAC_SECRET"] = eumdac_secret

    if os.environ.get("EUMDAC_KEY") is None or os.environ.get("EUMDAC_SECRET") is None:
        msg = "Please set your EUMDAC credentials as environment variables using:"
        msg += "\nexport EUMDAC_KEY=<your user key>"
        msg += "\nexport EUMDAC_SECRET=<your user secret>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--eumdac-key <your user key> --eumdac-secret <your user secret>"
        raise ValueError(msg)
    else:
        eumdac_key = os.environ.get("EUMDAC_KEY")
        eumdac_secret = os.environ.get("EUMDAC_SECRET")
    
    # check if credentials are valid
    credentials = (eumdac_key, eumdac_secret)
    try:
        token = eumdac.AccessToken(credentials)
        logger.info(f"EUMDAC login successful. Token '{token}' expires {token.expiration}")
        return token
    except:
        msg = "EUMDAC login failed."
        msg += "\nPlease check your credentials and set them as environment variables using:"
        msg += "\nexport EUMDAC_KEY=<your user key>"
        msg += "\nexport EUMDAC_SECRET=<your user secret>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--eumdac-key <your user key> --eumdac-secret <your user secret>"
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
    
def _compile_list_of_dates(timestamp_dict: dict, predefined_timestamps: List[str]) -> List[datetime]:
    if predefined_timestamps is not None:
        _check_predefined_timestamps(predefined_timestamps=predefined_timestamps)
        if type(predefined_timestamps[0]) is datetime:
            list_of_dates = predefined_timestamps
        elif type(predefined_timestamps[0]) is str:
            list_of_dates = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in predefined_timestamps]
        logger.info(f"Using predefined timestamps.")
    elif timestamp_dict['start_date'] is not None:
        # check start/end dates/times
        if timestamp_dict['end_date'] is None:
            end_date = timestamp_dict['start_date']
        else:
            end_date = timestamp_dict['end_date']
        # combine date and time information
        start_datetime_str = timestamp_dict['start_date'] + ' ' + timestamp_dict['start_time']
        end_datetime_str = end_date + ' ' + timestamp_dict['end_time']
        _check_datetime_format(start_datetime_str, end_datetime_str)
        # datetime conversion 
        start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
        end_datetime = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M:%S")
        _check_start_end_times(start_datetime=start_datetime, end_datetime=end_datetime)

        # define time step for data query                       
        if timestamp_dict['time_step'] is None: 
            time_step = '1:00:00'
            logger.info("No timedelta specified. Default is 1 hour.")
        else:
            time_step = timestamp_dict['time_step']
        _check_timedelta_format(time_delta=time_step)
        
        # convert str to datetime object
        hours, minutes, seconds = convert_str2time(time=time_step)
        time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        
        _check_timedelta(time_delta=time_delta)
        
        # Compile list of dates/times
        list_of_dates = np.arange(start_datetime, end_datetime, time_delta).astype(datetime)
        print('Times to check: ',list_of_dates[0], list_of_dates[-1])

        window_date = '1991-10-19' # Add arbitrary date to convert into proper datetime object
        start_datetime_window_str = window_date + ' ' + timestamp_dict['daily_window_t0']
        end_datetime_window_str = window_date + ' ' + timestamp_dict['daily_window_t1']
        _check_start_end_times(start_datetime=start_datetime, end_datetime=end_datetime)
        # datetime conversion 
        daily_window_t0_datetime = datetime.strptime(start_datetime_window_str, "%Y-%m-%d %H:%M:%S")
        daily_window_t1_datetime = datetime.strptime(end_datetime_window_str, "%Y-%m-%d %H:%M:%S")
        _check_start_end_times(start_datetime=daily_window_t0_datetime, end_datetime=daily_window_t1_datetime)

        # filter function - check that query times fall within desired time window
        def is_in_between(date):
            return daily_window_t0_datetime.time() <= date.time() <= daily_window_t1_datetime.time()

        # compile new list of dates within desired time window
        list_of_dates = list(filter(is_in_between, list_of_dates))
        logger.info("Compiling timestamps from specific parameters.")
    else:
        msg = "Please provide either predefined timestamps or start date"
        raise ValueError(msg)
    return list_of_dates

def _check_predefined_timestamps(predefined_timestamps: List[str]) -> bool:
    if type(predefined_timestamps) is not list:
        msg = "Please provide predefined timestamps as a list"
        raise ValueError(msg)
    if type(predefined_timestamps[0]) is str: # Check type of first element
        try:
            for x in predefined_timestamps:
                datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
            return True
        except Exception as e:
            msg = "Please check predefined timestamps"
            msg += "\nExpected date format: %Y-%m-%d"
            msg += "\nExpected time format: %H:%M:%S"
            raise SyntaxError(msg)
    elif type(predefined_timestamps[0]) is datetime: # Check type of first element
        return True
    else:
        msg = "Please check predefined timestamps"
        msg += "\nExpected either datetime objects or strings in the format %Y-%m-%d %H:%M:%S"
        raise SyntaxError(msg)

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
            

def _check_start_end_times(start_datetime: datetime, end_datetime: datetime) -> bool:
    """ check end_datetime is after start_datetime """
    if start_datetime < end_datetime:
        return True
    else:
        msg = "Start datetime must be before end datetime\n"
        msg += f"This does not hold for start = {str(start_datetime)} and end = {str(end_datetime)}"
        raise ValueError(msg)
    
def _check_timedelta_format(time_delta: str) -> bool:
    try:
        time_list = time_delta.split(":")
        assert len(time_list) == 3
        assert 0 <= int(time_list[0]) # Check that hours is >= 0, and convertible to int
        assert 0 <= int(time_list[1]) < 60 # Check that minutes < 60, and convertible to int
        assert 0 <= int(time_list[2]) < 60 # Check that seconds < 60, and convertible to int

    except Exception as e:
        msg = "Please check time step format"
        msg += "\nExpected time format: %H:%M:%S"
        raise SyntaxError(msg)

def _check_timedelta(time_delta: datetime) -> bool:
    if time_delta.days > 0: return True
    if time_delta.seconds >= 15 * 60: return True

    msg = "Time delta must not be smaller than the time resolution of the data\n"
    msg += f"Time delta {str(time_delta)} is too small\n"
    msg += f"The minimum required time delta is 15 min"
    raise ValueError(msg)

def _check_input_processing_level(processing_level: str) -> bool:
    """checks processing level for MSG data"""
    if processing_level in ["L1"]:
        return True
    else:
        msg = "Unrecognized processing level"
        msg += f"\nNeeds to be 'L1'. Others are not yet tested"
        raise ValueError(msg)

def _check_instrument(instrument: str) -> bool:
    """checks instrument for MSG data."""
    if instrument in ["HRSEVIRI", "CLM"]:
        return True
    else:
        msg = "Unrecognized instrument"
        msg += f"\nNeeds to be 'HRSEVIRI' or 'CLM'. Others are not yet tested"
        raise ValueError(msg)
    
def _check_data_product_name(data_product: str) -> bool:
    if data_product in ['EO:EUM:DAT:MSG:HRSEVIRI', 'EO:EUM:DAT:MSG:CLM']:
        return True
    else:
        msg = f"Unrecognized data product {data_product}"
        raise ValueError(msg)
             
def convert_str2time(time: str):
    time_list = time.split(":")
    hours = int(time_list[0])
    minutes = int(time_list[1])
    seconds = int(time_list[2])

    return hours, minutes, seconds

def delete_list_of_files(file_list: List[str]) -> None:
    for file_path in file_list:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")

def _check_save_dir(save_dir: str) -> bool:
    """ check if save_dir exists """
    if os.path.isdir(save_dir):
        return True
    else:
        try:
            os.makedirs(save_dir)
            return True
        except:
            msg = "Save directory does not exist"
            msg += f"\nReceived: {save_dir}"
            msg += "\nCould not create directory"
            raise ValueError(msg)

def main(input: str):

    print(input)

if __name__ == '__main__':
    typer.run(msg_download)

    """
    # =========================
    # MSG LEVEL 1 Test Cases
    # =========================
    # custom day
    python scripts/msg-download.py 2018-10-01
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01
    # custom day + end points
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 09:00:00 --end-time 12:00:00
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --start-time 09:05:00 --end-time 12:05:00
    # custom day + end points + time window
    scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 00:05:00 --end-time 23:54:00 --daily-window-t0 09:00:00 --daily-window-t1 12:00:00 
    # custom day + end points + time window + timestep
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 00:05:00 --end-time 23:54:00 --daily-window-t0 09:00:00 --daily-window-t1 12:00:00 --time-step 00:15:00
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-01 --start-time 00:05:00 --end-time 23:54:00 --daily-window-t0 09:00:00 --daily-window-t1 12:00:00 --time-step 00:25:00
    # ===================================
    # MSG CLOUD MASK Test Cases
    # ===================================
    # custom day
    python scripts/msg-download.py 2018-10-01 --instrument=CLM
    # custom day + end points
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --instrument=CLM 
    # custom day + end points + time window
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --start-time 09:00:00 --end-time 12:00:00 --instrument=CLM 
    # custom day + end points + time window + timestep
    python scripts/msg-download.py 2018-10-01 --end-date 2018-10-05 --start-time 09:00:00 --end-time 12:00:00 --time-step 00:25:00 --instrument=CLM
    """
