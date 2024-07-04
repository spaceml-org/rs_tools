from typing import Union, List, Dict, Tuple
import numpy as np
import pandas as pd
from loguru import logger


def match_timestamps(times_data: List[str], times_clouds: List[str], cutoff: int=15) -> pd.DataFrame:
    """
    Matches timestamps of data and cloudmask files, if not measured at exactly the same time.

    Args:
        times_data (List[str]): Timestamps of data files.
        times_clouds (List[str]): Timestamps of the cloud mask files.
        cutoff (str, optional): Maximum time difference in minutes to consider a match. Defaults to 15.

    Returns:
        pd.DataFrame: A DataFrame with the matched timestamps.
    """
    # Convert timestamps to datetime objects
    timestamps_data = pd.to_datetime(times_data)
    timestamps_clouds = pd.to_datetime(times_clouds)

    matches_data = []
    matches_clouds = []

    # Loop through timestamps of data files
    for time in timestamps_data:
        # Find closest timestamp in cloud mask files
        closest_time = timestamps_clouds[
            np.abs((timestamps_clouds - time).total_seconds()).argmin()
        ]
        # Check if the closest timestamp is within the cutoff time
        if np.abs((closest_time - time).total_seconds()) <= pd.Timedelta(f'{cutoff}min').total_seconds():
            matches_data.append(time.strftime("%Y%m%d%H%M%S"))
            matches_clouds.append(closest_time.strftime("%Y%m%d%H%M%S"))
        else:
            logger.info(f"No matching cloud mask found for {time}")

    matched_times = pd.DataFrame({
        "timestamps_data": matches_data,
        "timestamps_cloudmask": matches_clouds
    })
            
    return matched_times

