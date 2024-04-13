from typing import Optional, List, Union, Tuple
from omegaconf import DictConfig
from datetime import datetime
import pandas as pd

def split_dataset(filenames: List, split_spec: DictConfig) -> Tuple[List, List]:
    """
    Split filenames into training and validation sets based on dataset specification.

    Args:
        filenames (List): A list of filenames to be split.
        split_spec (DictConfig): A dictionary-like object containing the dataset specification.

    Returns:
        Tuple[List, List]: A tuple containing two lists: the training set and the validation set.
    """
    if "train" not in split_spec.keys() or "val" not in split_spec.keys():
        raise ValueError("split_spec must contain 'train' and 'val' keys")
    
    # Extract dates from filenames
    dates = get_dates_from_files(filenames)
    # Convert to dataframe for easier manipulation
    df = pd.DataFrame({"filename": filenames, "date": dates})

    train_split = split_spec["train"]
    val_split = split_spec["val"]

    # Check if years, months, and days are specified
    # If not, use all available years/months/days
    if train_split["years"] is None:
        train_split["years"] = df.date.dt.year.unique().tolist()
    if val_split["years"] is None:
        val_split["years"] = df.date.dt.year.unique().tolist()
    if train_split["months"] is None:
        train_split["months"] = df.date.dt.month.unique().tolist()
    if val_split["months"] is None:
        val_split["months"] = df.date.dt.month.unique().tolist()
    if train_split["days"] is None:
        train_split["days"] = df.date.dt.day.unique().tolist()
    if val_split["days"] is None:
        val_split["days"] = df.date.dt.day.unique().tolist()

    # Determine conditions for training and validation sets
    train_condition = (df.date.dt.year.isin(train_split["years"])) & \
                        (df.date.dt.month.isin(train_split["months"])) & \
                        (df.date.dt.day.isin(train_split["days"]))
    val_condition = (df.date.dt.year.isin(val_split["years"])) & \
                        (df.date.dt.month.isin(val_split["months"])) & \
                        (df.date.dt.day.isin(val_split["days"]))
    
    # Extract filenames based on conditions
    train_files = df[train_condition].filename.tolist()
    val_files = df[val_condition].filename.tolist()

    # Check if files are allocated properly
    if len(train_files) == 0:
        raise ValueError("No training files found. Check split specification.")
    if len(val_files) == 0:
        raise ValueError("No validation files found. Check split specification.")
    if len(train_files) + len(val_files) > len(filenames):
        raise ValueError("Duplicate file allocation. Check split specification.")

    return train_files, val_files

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



