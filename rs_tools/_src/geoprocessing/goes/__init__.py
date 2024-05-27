from datetime import datetime, timedelta
from pathlib import Path

# All wavelengths in micrometers
GOES_WAVELENGTHS = {
    "1": 0.47,
    "2": 0.64,
    "3": 0.87,
    "4": 1.38,
    "5": 1.61,
    "6": 2.25,
    "7": 3.89,
    "8": 6.17,
    "9": 6.93,
    "10": 7.34,
    "11": 8.44,
    "12": 9.61,
    "13": 10.33,
    "14": 11.19,
    "15": 12.27,
    "16": 13.27,
}

GOES_CHANNELS = {
    0.47: 1,
    0.64: 2,
    0.87: 3,
    1.38: 4,
    1.61: 5,
    2.25: 6,
    3.89: 7,
    6.17: 8,
    6.93: 9,
    7.34: 10,
    8.44: 11,
    9.61: 12,
    10.33: 13,
    11.19: 14,
    12.27: 15,
    13.27: 16,
}   

def parse_goes16_dates_from_file(file: str):
    """
    Parses the date and time information from a GOES-16 file name.

    Args:
        file (str): The file name to parse.

    Returns:
        str: The parsed date and time in the format 'YYYYJJJHHMM'.
    """
    timestamp = Path(file).name.replace("-", "_").split("_")
    
    return datetime.strptime(timestamp[-3][1:], "%Y%j%H%M%S%f").strftime("%Y%j%H%M%S")

def format_goes_dates(time: str) -> str:
    """
    Function to format the date/time string.
    
    Args:
        time (str): The time string to be formatted.
        
    Returns:
        str: The formatted time string.
    """

    # Split the string into date and time parts
    date_str, time_str = time[:7], time[7:]
    # Convert the date part to a datetime object
    date = datetime.strptime(date_str, "%Y%j")

    # Convert the time part to a timedelta object
    time = timedelta(hours=int(time_str[:2]), minutes=int(time_str[2:4]), seconds=int(time_str[4:6]))
    # Add the date and time parts to get a datetime object
    dt = date + time
    # Convert the datetime object to a string in the format "YYYYMMDDHHMMSS"
    str_time = dt.strftime("%Y%m%d%H%M%S")
    
    return str_time