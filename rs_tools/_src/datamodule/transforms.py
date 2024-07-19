from typing import List, Dict, Optional, Callable
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class DictionaryTransformer:
    """
    A class that transforms a dictionary by applying a function to a specific key's value.

    Args:
        input_key (str): The key in the input dictionary whose value will be transformed.
        fn (Callable): The transformation function to be applied to the value.
        output_key (Optional[str]): The key in the output dictionary where the transformed value will be stored.
            If not provided, the transformed value will replace the original value at the input key.

    Methods:
        __call__(self, input: Dict[str, NDArray]) -> Dict[str, NDArray]:
            Applies the transformation function to the value at the input key and returns the modified dictionary.

    Attributes:
        input_key (str): The key in the input dictionary whose value will be transformed.
        fn (Callable): The transformation function to be applied to the value.
        output_key (Optional[str]): The key in the output dictionary where the transformed value will be stored.
    """

    input_key: str
    fn: Callable
    output_key: Optional[str] = None

    def __post_init__(self):
        self.output_key = self.input_key if self.output_key is None else self.output_key
    
    def __call__(self, input: Dict[str, NDArray]) -> Dict[str, NDArray]:
        """
        Applies the transformation function to the value at the input key and returns the modified dictionary.

        Args:
            input (Dict[str, NDArray]): The input dictionary.

        Returns:
            Dict[str, NDArray]: The modified dictionary with the transformed value.
        """
        # select data
        data = input[self.input_key]
        # apply transformation
        data = self.fn(data)
        # update dictionary
        input[self.output_key] = data
        # return modified datastructure
        return input


def parse_band_indices(
    band_queries: List[str],
    band_names: List[str],
) -> Dict[[str], int]:

    # create a dictionary with the band name and indices
    band_names_dict = {iname: index+1 for index, iname in enumerate(band_names)}
    # select bands from input band names
    band_names_dict = {k: band_names_dict[k] for k in band_queries if k in band_names_dict}
    return band_names_dict
    

def transform_band_order():
    raise NotImplementedError()


def transform_band_selection():
    raise NotImplementedError()


import numpy as np

def transform_nan_mask(array):
    """
    Transform NaN mask.

    This function takes an input array and returns a binary mask indicating the presence of NaN values in the array.

    Parameters:
    array (numpy.ndarray): Input array.

    Returns:
    numpy.ndarray: Binary mask indicating the presence of NaN values in the input array.
    """
    mask = np.isnan(array).any(axis=0)
    mask = mask.astype(int)
    return mask


def nan_fill(array, fill_value: float=0.0):
    """
    Fill NaN values in the input array with the specified fill value.

    Parameters:
        array (numpy.ndarray): The input array.
        fill_value (float, optional): The value to fill NaN values with. Defaults to 0.0.

    Returns:
        numpy.ndarray: The array with NaN values filled.

    """
    return np.nan_to_num(array, nan=fill_value)


def normalize_latlon(array):
    """
    Normalize latitude values in the given array.

    Parameters:
    array (numpy.ndarray): The input array containing latitude values.

    Returns:
    numpy.ndarray: The array with normalized latitude values.

    """
    lats, lons = array
    lats /= 90
    lons /= 180
    array = np.stack([lats, lons], axis=0)
    return array


def transform_radiance_units(array):
    """
    Transforms the given array from radiance units to a normalized scale.

    Args:
        array (numpy.ndarray): The input array representing radiance values.

    Returns:
        numpy.ndarray: The transformed array with radiance values normalized to a scale of 0 to 1.
    """
    return array / 180


def stack_dictionary(array):
    raise NotImplementedError()