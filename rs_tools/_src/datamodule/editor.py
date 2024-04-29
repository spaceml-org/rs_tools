import torch
import numpy as np

from iti.data.editor import Editor
from rs_tools._src.geoprocessing.units import convert_units

from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)

# NOTE: Code already moved to ITI repo
class EditorPatch(Editor):
    def __call__(self, data_dict, **kwargs):
        return self.call(data_dict, **kwargs)

class BandOrderEditor(EditorPatch):
    """
    Reorders bands in data dictionary.
    """

    def __init__(self, target_order, key="data"):
        """
        Args:
            target_order (list): Order of bands
            key (str): Key in dictionary to apply transformation
        """
        self.target_order = target_order
        self.key = key

    def call(self, data_dict, **kwargs):
        source_order = data_dict["wavelengths"]
        assert len(source_order) == len(self.target_order), "Length of source and target wavelengths must match."
        # Get indexes of bands to select
        indexes = [np.where(source_order == wvl)[0][0] for wvl in self.target_order]
        # Extract data
        data = data_dict[self.key]
        # Subselect bands
        data = data[indexes]
        # Update dictionary
        data_dict[self.key] = data
        data_dict["wavelengths"] = np.array(self.target_order)
        return data_dict

class BandSelectionEditor(EditorPatch):
    """
    Selects a subset of available bands from data dictionary
    """
    def __init__(self, target_bands, key="data"):
        """
        Args:
            target_bands (list): List of bands to select
            key (str): Key in dictionary to apply transformation
        """
        self.target_bands = target_bands
        self.key = key

    def call(self, data_dict, **kwargs):
        source_bands = data_dict["wavelengths"]
        # Get indexes of bands to select
        indexes = [np.where(source_bands == wvl)[0][0] for wvl in self.target_bands]
        # Extract data
        data = data_dict[self.key]
        # Subselect bands
        data = data[indexes]
        assert data.shape[0] == len(self.target_bands)
        # Update dictionary
        data_dict[self.key] = data
        data_dict["wavelengths"] = np.array(self.target_bands)
        return data_dict

class NanMaskEditor(EditorPatch):
    """
    Returns mask for NaN values in data dictionary
    """
    def __init__(self, key="data"):
        self.key = key
    def call(self, data_dict, **kwargs):
        data = data_dict[self.key]
        # Check if any band contains NaN values
        mask = np.isnan(data).any(axis=0)
        mask = mask.astype(int)
        # Update dictionary
        data_dict["nan_mask"] = mask
        return data_dict

class NanDictEditor(EditorPatch):
    """
    Removes NaN values from data dictionary.
    Can also be used to replace NaN values of coordinates to remove off limb data.
    """
    def __init__(self, key="data", fill_value=0):
        self.key = key
        self.fill_value = fill_value
    def call(self, data_dict, **kwargs):
        data = data_dict[self.key]
        # Replace NaN values
        data = np.nan_to_num(data, nan=self.fill_value)
        # Update dictionary
        data_dict[self.key] = data
        return data_dict
    
class CoordNormEditor(EditorPatch):
    """
    Normalize latitude and longitude coordinates
    """
    def __init__(self, key="coords"):
        self.key = key
    def call(self, data_dict, **kwargs):
        lats, lons = data_dict["coords"]
        # Normalize latitude and longitude to range [-1, 1]
        lats = lats/90
        lons = lons/180
        # Update dictionary
        data_dict["coords"] = np.stack([lats, lons], axis=0)
        return data_dict
     
class RadUnitEditor(EditorPatch):
    """
    Convert radiance values from mW/m^2/sr/cm^-1 to W/m^2/sr/um
    """
    def __init__(self, key="data"):
        self.key = key
    def call(self, data_dict, **kwargs):
        data = data_dict[self.key]
        wavelengths = data_dict["wavelengths"]
        # Convert units
        data = convert_units(data, wavelengths)
        # Update dictionary
        data_dict[self.key] = data
        return data_dict
    
class StackDictEditor(EditorPatch):
    """
    Stack data dictionary into a single array
    """
    def __init__(self, allowed_keys=["data", "cloud_mask", "nan_mask", "coords"], axis=0):
        self.allowed_keys = allowed_keys
        self.axis = axis
    def call(self, data_dict, **kwargs):
        # Select keys
        self.keys = [key for key in self.allowed_keys if key in data_dict.keys()]
        # Select data
        data = []
        for key in self.keys:
            values = data_dict[key]
            if len(values.shape) == 2:
                values = np.expand_dims(values, axis=self.axis)
            data.append(values)
        # Stack data
        data = np.concatenate(data, axis=self.axis)
        # Return numpy array
        return data
    
class ToTensorEditor(EditorPatch):
    """
    Convert numpy array to PyTorch tensor
    """
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    def call(self, data, **kwargs):
        # Convert to tensor
        tensor = torch.as_tensor(data, dtype=self.dtype)
        return tensor



