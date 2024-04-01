from pathlib import Path
from datetime import datetime, timedelta

MODIS_1KM_VARIABLES = {
    "EV_1KM_Emissive": [20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36],
    "EV_1KM_RefSB": [8,9,10,11,12,"13lo","13hi","14lo","14hi",15,16,17,18,19,26],
    "EV_250_Aggr1km_RefSB": [1,2],
    "EV_500_Aggr1km_RefSB": [3,4,5,6,7],
}

# All given in micrometers
# Central wavelength calculated as mean of range
MODIS_WAVELENGTHS = {
    "1": 0.645, 
    "2": 0.8585,
    "3": 0.469,
    "4": 0.555,
    "5": 1.24,
    "6": 1.64,
    "7": 2.13,
    "8": 0.4125,
    "9": 0.443,
    "10": 0.488,
    "11": 0.531,
    "12": 0.551,
    "13lo": 0.667,
    "13hi": 0.667,
    "14lo": 0.678,
    "14hi": 0.678,
    "15": 0.748,
    "16": 0.8695,
    "17": 0.905,
    "18": 0.936,
    "19": 0.940,
    "20": 3.75,
    "21": 3.959,
    "22": 3.959,
    "23": 4.05,
    "24": 4.4655,
    "25": 4.5155,
    "26": 1.375,
    "27": 6.715,
    "28": 7.325,
    "29": 8.55,
    "30": 9.73,
    "31": 11.03,
    "32": 12.02,
    "33": 13.335,
    "34": 13.635,
    "35": 13.935,
    "36": 14.235,
}

def parse_modis_dates_from_file(file: str):
    """
    Parses the date and time information from a MODIS file name.

    Args:
        file (str): The file name to parse.

    Returns:
        str: The parsed date and time in the format 'YYYYJJJHHMM'.
    """
    # get the date from the file
    date = Path(file).name.split(".")[1][1:]
    # get the time from the file
    time = Path(file).name.split(".")[2]
    datetime_str = f"{date}.{time}"

    return datetime_str

def format_modis_dates(time: str) -> str:
    """
    Function to format the date/time string.
    
    Args:
        time (str): The time string to be formatted.
        
    Returns:
        str: The formatted time string.
    """
    # Split the string into date and time parts
    date_str, time_str = time.split(".")
    # Convert the date part to a datetime object
    date = datetime.strptime(date_str, "%Y%j")
    # Convert the time part to a timedelta object
    time = timedelta(hours=int(time_str[:2]), minutes=int(time_str[2:]))
    # Add the date and time parts to get a datetime object
    dt = date + time
    # Convert the datetime object to a string in the format "YYYYMMDDHHMMSS"
    str_time = dt.strftime("%Y%m%d%H%M%S")

    return str_time
