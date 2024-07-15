import numpy as np


def check_nan_count(arr: np.array, nan_cutoff: float) -> bool:
    """
    Check if the number of NaN values in the given array is below a specified cutoff.

    Parameters:
        arr (np.array): The input array to check for NaN values.
        nan_cutoff (float): The maximum allowed ratio of NaN values to the total number of values.

    Returns:
        bool: True if the number of NaN values is below the cutoff, False otherwise.
    """
    # count nans in dataset
    nan_count = int(np.count_nonzero(np.isnan(arr)))
    # get total pixel count
    total_count = int(arr.size)
    # check if nan_count is within allowed cutoff
    if nan_count/total_count <= nan_cutoff:
        return True
    else:
        return False