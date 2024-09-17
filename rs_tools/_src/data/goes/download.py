from typing import Optional, List, Union
import os
import warnings
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import typer
from loguru import logger
from datetime import datetime, timedelta

from goes2go import GOES
from goes2go.data import goes_nearesttime

# The cadence depends on the measurement scale
# The full disk is measured every 15 mins
# CONUS is measured every 5 mins
# Mesoscale is measured every 1 min
DOMAIN_TIMESTEP = {
    'F': 15,
    'C': 5,
    'M': 1
}

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# Your code here

def goes_download(
    start_date: Optional[str]='2018-10-01', # TODO: Wrap date/time parameters in dict to reduce arguments?
    end_date: Optional[str]=None,
    start_time: Optional[str]='00:00:00',
    end_time: Optional[str]='23:59:00',
    daily_window_t0: Optional[str]='00:00:00',
    daily_window_t1: Optional[str]='23:59:00',
    time_step: Optional[str]=None,
    predefined_timestamps: Optional[List[str]]=None,
    satellite_number: int=16,
    save_dir: Optional[str] = ".",
    instrument: str = "ABI",
    processing_level: str = 'L1b',
    data_product: str = 'Rad',
    domain: str = 'F',
    bands: Optional[str] = "all",
    check_bands_downloaded: bool = False,
):
    """
    Downloads GOES satellite data for a specified time period and set of bands.

    Args:
        start_date (str, optional): The start date of the data download in the format 'YYYY-MM-DD'.
        end_date (str, optional): The end date of the data download in the format 'YYYY-MM-DD'. If not provided, the end date will be the same as the start date.
        start_time (str, optional): The start time of the data download in the format 'HH:MM:SS'. Default is '00:00:00'.
        end_time (str, optional): The end time of the data download in the format 'HH:MM:SS'. Default is '23:59:00'.
        daily_window_t0 (str, optional): The start time of the daily window in the format 'HH:MM:SS'. Default is '00:00:00'. Used if e.g., only day/night measurements are required.
        daily_window_t1 (str, optional): The end time of the daily window in the format 'HH:MM:SS'. Default is '23:59:00'. Used if e.g., only day/night measurements are required.
        time_step (str, optional): The time step between each data download in the format 'HH:MM:SS'. If not provided, the default is 1 hour.
        predefined_timestamps (list, optional): A list of timestamps to download. Expected format is datetime or str following 'YYYY-MM-DD HH:MM:SS'. If provided, start/end dates/times will be ignored.
        satellite_number (int, optional): The satellite number. Default is 16.
        save_dir (str, optional): The directory where the downloaded files will be saved. Default is the current directory.
        instrument (str, optional): The instrument name. Default is 'ABI'.
        processing_level (str, optional): The processing level of the data. Default is 'L1b'.
        data_product (str, optional): The data product to download. Default is 'Rad'.
        domain (str, optional): The domain of the data. Default is 'F' - Full Disk.
        bands (str, optional): The bands to download. Default is 'all'.
        check_bands_downloaded (bool, optional): Whether to check if all bands were successfully downloaded for each time step. Default is False.

    Returns:
        list: A list of file paths for the downloaded files.
        
    Examples:
        # =========================
        # GOES LEVEL 1B Test Cases
        # =========================
        # custom day
        python scripts/goes-download.py 2020-10-01 --end-date 2020-10-01
        # custom day + end points
        python scripts/goes-download.py 2020-10-01 --end-date 2020-10-01 --start-time 00:00:00 --end-time 23:00:00
        # custom day + end points + time window
        python scripts/goes-download.py 2020-10-01 --end-date 2020-10-01 --start-time 00:00:00 --end-time 23:00:00 --daily-window-t0 08:30:00 --daily-window-t1 21:30:00
        # custom day + end points + time window + timestep
        python scripts/goes-download.py 2020-10-01 --end-date 2020-10-01 --start-time 00:00:00 --end-time 23:00:00 --daily-window-t0 08:30:00 --daily-window-t1 21:30:00 --time-step 06:00:00
        # ===================================
        # GOES LEVEL 2 CLOUD MASK Test Cases
        # ===================================
        python scripts/goes-download.py 2020-10-01 --start-time 10:00:00 --end-time 11:00:00 --processing-level L2 --data-product ACM
        
        # ====================
        # FAILURE TEST CASES
        # ====================
        python scripts/goes-download.py 2018-10-01 --end-date 2018-10-01 --daily-window-t0 17:00:00 --daily-window-t1 17:14:00 --time-step 00:15:00 --save-dir /home/juanjohn/data/
        python scripts/goes-download.py 2018-10-01 --end-date 2018-10-01 --daily-window-t0 17:00:00 --daily-window-t1 17:14:00 --time-step 00:15:00 --save-dir /home/juanjohn/data/ --check-bands-downloaded
    """
    # run checks
    # check satellite details
    _check_input_processing_level(processing_level=processing_level)
    _check_instrument(instrument=instrument)
    _check_satellite_number(satellite_number=satellite_number)
    logger.info(f"Satellite Number: {satellite_number}")
    _check_domain(domain=domain)
    # compile bands
    if processing_level == 'L1b':
        list_of_bands = _check_bands(bands=bands)
    elif processing_level == 'L2':
        list_of_bands = None
    else:
        raise ValueError('bands not correctly specified for given processing level')
    # check data product
    data_product = f"{instrument}-{processing_level}-{data_product}{domain}"
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
        'domain': domain,
    }

    # compile list of dates
    list_of_dates = _compile_list_of_dates(timestamp_dict=timestamp_dict, predefined_timestamps=predefined_timestamps)

    # check if save_dir is valid before attempting to download
    _check_save_dir(save_dir=save_dir)

    files = []

    # create progress bars for dates and bands
    pbar_time = tqdm.tqdm(list_of_dates)

    for itime in pbar_time:
        
        pbar_time.set_description(f"Time - {itime}")
        
        if processing_level == 'L1b':
            sub_files_list = _goes_level1_download(
                time=itime, 
                list_of_bands=list_of_bands,
                satellite_number=satellite_number,
                data_product=data_product,
                domain=domain,
                save_dir=save_dir,
                check_bands_downloaded=check_bands_downloaded,
                )
        elif processing_level == 'L2':
            sub_files_list = _goes_level2_download(
                time=itime, 
                satellite_number=satellite_number,
                data_product=data_product,
                domain=domain,
                save_dir=save_dir)
        else:
            raise ValueError(f"Unrecognized processing level: {processing_level}")

        files += sub_files_list

    return files


def _goes_level2_download(time,
                      satellite_number,
                      data_product,
                      domain,
                      save_dir):                  
                           
    try:
        ifile: pd.DataFrame = goes_nearesttime(
            attime=time,
            within=pd.to_timedelta(15, 'm'),
            satellite=satellite_number, 
            product=data_product, 
            domain=domain, 
            return_as="filelist", 
            save_dir=save_dir,
        )
        # extract filepath from GOES download pandas dataframe
        filepath: str = os.path.join(save_dir, ifile.file[0])
        return [filepath]
    except IndexError:
        logger.info(f"Data could not be downloaded for time step {time}.")
        return []
    
def _goes_level1_download(time, 
                      list_of_bands,
                      satellite_number, 
                      data_product, 
                      domain, 
                      save_dir,
                      check_bands_downloaded
                      ):

    sub_files_list: list[str] = []
    pbar_bands = tqdm.tqdm(list_of_bands)


    for iband in pbar_bands:
        
        pbar_bands.set_description(f"Band - {iband}")
        # download file
        try:
            ifile: pd.DataFrame = goes_nearesttime(
                attime=time,
                within=pd.to_timedelta(15, 'm'),
                satellite=satellite_number, 
                product=data_product, 
                domain=domain, 
                bands=iband, 
                return_as="filelist", 
                save_dir=save_dir,
            )
            # extract filepath from GOES download pandas dataframe
            filepath: str = os.path.join(save_dir, ifile.file[0])
            sub_files_list += [filepath]
        
        except IndexError:
            logger.info(f"Band {iband} could not be downloaded for time step {time}.")
            if check_bands_downloaded:
                logger.info(f"Deleting all other bands for time step {time}.")
                delete_list_of_files(sub_files_list) # delete partially downloaded bands
                return []

    return sub_files_list

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
        
        _check_timedelta(time_delta=time_delta, domain=timestamp_dict['domain'])
        
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

def _check_timedelta(time_delta: datetime, domain: str) -> bool:
    if time_delta.days > 0: return True
    
    if time_delta.seconds >= DOMAIN_TIMESTEP[domain] * 60: return True

    msg = "Time delta must not be smaller than the time resolution of the data\n"
    msg += f"Time delta {str(time_delta)} is too small for domain {domain}\n"
    msg += f"The minimum required time delta is {DOMAIN_TIMESTEP[domain]} min"
    raise ValueError(msg)

def _check_domain(domain: str) -> bool:
    """checks domain GOES data"""
    if str(domain) in ["F", "C", "M"]:
        return True
    else:
        msg = "Unrecognized domain"
        msg += f"\nNeeds to be 'F', 'C', 'M'."
        msg += "\nOthers are not yet implemented"
        raise ValueError(msg)
    

def _check_satellite_number(satellite_number: int) -> bool:
    """checks satellite number for GOES data"""
    if satellite_number in [16, 17, 18]:
        return True
    else:
        msg = "Unrecognized satellite number level"
        msg += f"\nNeeds to be 16, 17, or 18."
        msg += "\nOthers are not yet implemented"
        msg += f"\nInput: {satellite_number}"
        raise ValueError(msg)
    

def _check_input_processing_level(processing_level: str) -> bool:
    """checks processing level for GOES data"""
    if processing_level in ["L1b", "L2"]:
        return True
    else:
        msg = "Unrecognized processing level"
        msg += f"\nNeeds to be 'L1b' or 'L2'. Others are not yet tested"
        raise ValueError(msg)


def _check_instrument(instrument: str) -> bool:
    """checks instrument for GOES data."""
    if instrument in ["ABI"]:
        return True
    else:
        msg = "Unrecognized instrument"
        msg += f"\nNeeds to be 'ABI'. Others are not yet tested"
        raise ValueError(msg)
    
def _check_data_product_name(data_product: str) -> bool:
    if data_product in ['ABI-L1b-RadF', 'ABI-L1b-RadM', 'ABI-L1b-RadC', 'ABI-L1b-Rad',
                        'ABI-L2-ACMF', 'ABI-L2-ACMM', 'ABI-L2-ACMC']:
        return True
    else:
        msg = f"Unrecognized data product {data_product}"
        raise ValueError(msg)
        
def _check_bands(bands: str) -> List[int]:
    if bands in ['all']:
        list_of_bands = list(np.arange(1, 17))
        return list_of_bands
    else:
        try:
            list_of_bands = list(set(map(int, bands.split(' '))))
            logger.debug(f"List of str Bands to Ints: {list_of_bands}")
    
            criteria = lambda x: 17 > x > 0
            result = list(map(criteria, list_of_bands))
            logger.debug(f"Result from criteria: {result}")
    
            assert sum(result) == len(list_of_bands)
            return list_of_bands
        except Exception as e:
            msg = "Unrecognized bands"
            msg += f"\nNeeds to be 'all' or string of valid bands separated by spaces"
            msg += '\n(e.g., "13 14", \'1 2 3\').'
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
    typer.run(goes_download)

    """
    # custom day
    python rs_tools/scripts/goes-download.py 2020-10-01 --end-date 2020-10-01
    # custom day + end points
    python rs_tools/scripts/goes-download.py 2020-10-01 --end-date 2020-10-01 --start-time 00:00:00 --end-time 23:00:00
    # custom day + end points + time window
    python rs_tools/scripts/goes-download.py 2020-10-01 --end-date 2020-10-01 --start-time 00:00:00 --end-time 23:00:00 --daily-window-t0 08:30:00 --daily-window-t1 21:30:00
    # custom day + end points + time window + time step
    python rs_tools/scripts/goes-download.py 2020-10-01 --end-date 2020-10-01 --start-time 00:00:00 --end-time 23:00:00 --daily-window-t0 08:30:00 --daily-window-t1 21:30:00 --time-step 06:00:00
    # ====================
    # FAILURE TEST CASES
    # ====================
    python scripts/goes-download.py 2018-10-01 --end-date 2018-10-01 --daily-window-t0 17:00:00 --daily-window-t1 17:14:00 --time-step 00:15:00 --save-dir /home/juanjohn/data/
    python scripts/goes-download.py 2018-10-01 --end-date 2018-10-01 --daily-window-t0 17:00:00 --daily-window-t1 17:14:00 --time-step 00:15:00 --save-dir /home/juanjohn/data/ --check-bands-downloaded
    """
