"""
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

def goes_download(
    start_date: str,
    end_date: Optional[str]=None,
    start_time: Optional[str]='00:00:00',
    end_time: Optional[str]='23:59:00',
    time_step: Optional[str]=None,
    satellite_number: int=16,
    save_dir: Optional[str]=".",
    instrument: str = 'ABI',
    processing_level: str = 'L1b',
    data_product: str = 'Rad',
    domain: str = 'F',
    bands: str = "all",
    daytime_only: bool = False,
):


    # run checks
    _check_input_processing_level(processing_level=processing_level)
    _check_instrument(instrument=instrument)
    _check_satellite_number(satellite_number=satellite_number)
    logger.info(f"Satellite Number: {satellite_number}")
    _check_domain(domain=domain)
    # compile bands
    list_of_bands = _check_bands(bands=bands)
    # check data product
    data_product = f"{instrument}-{processing_level}-{data_product}"
    logger.info(f"Data Product: {data_product}")
    _check_data_product_name(data_product=data_product)

    # check start/end dates/times
    if end_date is None:
        end_date = start_date

    start_datetime_str = start_date + ' ' + start_time
    end_datetime_str = end_date + ' ' + end_time
    _check_datetime_format(start_datetime_str, end_datetime_str)
    # datetime conversion 
    start_datetime = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M:%S")
    end_datetime = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M:%S")

    _check_start_end_times(start_datetime=start_datetime, end_datetime=end_datetime)
                           
    if time_step is None: 
        time_step = '1:00:00'
        logger.info("No timedelta specified. Default is 1 hour.")
    _check_timedelta_format(time_delta=time_step)
    
    hours, minutes, seconds = convert_str2time(time=time_step)
    time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    _check_timedelta(time_delta=time_delta, domain=domain)
    
    # Compile list of dates/times
    list_of_dates = np.arange(start_datetime, end_datetime, time_delta).astype(datetime).astype(str)

    files = []

    # create progress bars for dates and bands
    pbar_time = tqdm.tqdm(list_of_dates)
    pbar_bands = tqdm.tqdm(list_of_bands)

    for itime in pbar_time:
        pbar_time.set_description(f"Time - {itime}")
        for iband in pbar_bands:
            pbar_bands.set_description(f"Band - {iband}")

            # ignore nighttime if user wants to
            if daytime_only:
                # TODO: check that centroid / radius points is inside daytime
                pass

            # download file
            logger.info(f"Bands: {iband}")
            ifile: list[str] = goes_nearesttime(
                attime=itime,
                within=pd.to_timedelta(15, 'm'),
                satellite=satellite_number, 
                product=data_product, 
                domain=domain, 
                bands=iband, 
                return_as="filelist", 
                save_dir=save_dir
            )
            # append list of files to larger list of files
            files.append(ifile)

            # TODO: check if all bands exist, otherwise skip to next timesttep
            # 

            # TODO: Add functions to process data
            
            # - open file
            # - change coordinate systems
            # - resample  (Change Period)
            # - rregrid
            
            break
        break

    return files


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
    # TODO: Check mesoscale
    if str(domain) in ["F", "C", "M"]:
        return True
    else:
        msg = "Unrecognized domain"
        msg += f"\nNeeds to be 'F', 'C', 'M'."
        msg += "\nOthers are not yet implemented"
        raise ValueError(msg)
    

def _check_satellite_number(satellite_number: str) -> bool:
    """checks satellite number for GOES data"""
    if str(satellite_number) in ["16", "17", "18"]:
        return True
    else:
        msg = "Unrecognized satellite number level"
        msg += f"\nNeeds to be '16', '17', or '18'."
        msg += "\nOthers are not yet implemented"
        raise ValueError(msg)
    

def _check_input_processing_level(processing_level: str) -> bool:
    """checks processing level for GOES datas"""
    if processing_level in ["L1b"]:
        return True
    else:
        msg = "Unrecognized processing level"
        msg += f"\nNeeds to be 'L1b'. Others are not yet implemented"
        raise ValueError(msg)


def _check_instrument(instrument: str) -> bool:
    """checks instrument for GOES data."""
    if instrument in ["ABI"]:
        return True
    else:
        msg = "Unrecognized instrument"
        msg += f"\nNeeds to be 'ABI'. Others are not yet implemented"
        raise ValueError(msg)
    
def _check_data_product_name(data_product: str) -> bool:
    if data_product in ['ABI-L1b-Rad']:
        return True
    else:
        msg = "Unrecognized data product"
        msg += f"\nNeeds to be 'ABI-L1b-Rad'. Others are not yet implemented"
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

def main(input: str):

    print(input)

if __name__ == '__main__':
    typer.run(goes_download)

    """
    python rs_tools/scripts/goes-download.py --bands "12 13"
    """
