from typing import Optional, List, Union, Tuple
from omegaconf import DictConfig
from datetime import datetime
import pandas as pd
from loguru import logger
import xarray as xr
import numpy as np

# NOTE: Code already moved to ITI repo
def split_train_val(files: List, split_spec: DictConfig) -> Tuple[List, List]:
    """
    Split files into training and validation sets based on dataset specification.

    Args:
        files (List): A list of files to be split.
        split_spec (DictConfig): A dictionary-like object containing the dataset specification.

    Returns:
        Tuple[List, List]: A tuple containing two lists: the training set and the validation set.
    """
    if "train" not in split_spec.keys() or "val" not in split_spec.keys():
        raise ValueError("split_spec must contain 'train' and 'val' keys")
    
    train_files = get_split(files, split_spec["train"])
    val_files = get_split(files, split_spec["val"])

    return train_files, val_files
    
    
def get_split(files: List, 
              split_dict: DictConfig) -> Tuple[List, List]:
    """
    Split files based on dataset specification.

    Args:
        files (List): A list of files to be split.
        split_dict (DictConfig): A dictionary-like object containing the dataset specification.

    Returns:
        Tuple[List, List]: A tuple containing two lists: the training set and the validation set.
    """
    # Extract dates from filenames
    filenames = [file.split("/")[-1] for file in files]
    dates = get_dates_from_files(filenames)
    # Convert to dataframe for easier manipulation
    df = pd.DataFrame({"filename": filenames, "files": files, "date": dates})

    # Check if years, months, and days are specified
    if "years" not in split_dict.keys() or split_dict["years"] is None:
        logger.info("No years specified for split. Using all years.")
        split_dict["years"] = df.date.dt.year.unique().tolist()
    if "months" not in split_dict.keys() or split_dict["months"] is None:
        logger.info("No months specified for split. Using all months.")
        split_dict["months"] = df.date.dt.month.unique().tolist()
    if "days" not in split_dict.keys() or split_dict["days"] is None:
        logger.info("No days specified for split. Using all days.")
        split_dict["days"] = df.date.dt.day.unique().tolist()

    # Determine conditions specified split
    condition = (df.date.dt.year.isin(split_dict["years"])) & \
                (df.date.dt.month.isin(split_dict["months"])) & \
                (df.date.dt.day.isin(split_dict["days"]))
        
    # Extract filenames based on conditions
    split_files = df[condition].files.tolist()

    # Check if files are allocated properly
    if len(split_files) == 0:
        raise ValueError("No files found. Check split specification.")
    
    return split_files

def get_date_from_file(filename: str) -> datetime:
    """
    Extract date from filename.

    Args:
        filenames (List[str]): A list of filenames.

    Returns:
        List[str]: A list of dates extracted from the filenames.
    """
    date = datetime.strptime(filename.split("_")[0], "%Y%m%d%H%M%S")
    return date

def get_dates_from_files(filenames: List[str]) -> List[datetime]:
    """
    Extract dates from a list of filenames.

    Args:
        filenames (List[str]): A list of filenames.

    Returns:
        List[str]: A list of dates extracted from the filenames.
    """
    dates = [datetime.strptime(filename.split("_")[0], "%Y%m%d%H%M%S") for filename in filenames]
    return dates


def load_nc_file(
    file: str, 
    load_coords: bool=True,
    load_cloudmask: bool=True,
):
    data_dict = {}
    # Load dataset
    ds: xr.Dataset = xr.load_dataset(file, engine="netcdf4")

    # Extract data
    data = ds.Rad.compute().to_numpy()
    data_dict["data"] = data
    # Extract wavelengths
    wavelengths = ds.band_wavelength.compute().to_numpy()
    data_dict["wavelengths"] = wavelengths

    # Extract coordinates
    if load_coords:
        latitude = ds.latitude.compute().to_numpy()
        longitude = ds.longitude.compute().to_numpy()
        coords = np.stack([latitude, longitude], axis=0)
        data_dict["coords"] = coords

    # Extract cloud mask
    if load_cloudmask:
        cloud_mask = ds.cloud_mask.compute().to_numpy()
        data_dict["cloud_mask"] = cloud_mask

    return data_dict



