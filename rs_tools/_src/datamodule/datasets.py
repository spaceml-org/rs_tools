from __future__ import annotations

import gc
import glob
import logging
import os
import random
from collections import Iterable
from typing import List, Union, Dict

import numpy as np
from dateutil.parser import parse
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
import numpy as np
from typing import List
from loguru import logger
from torch.utils.data import Dataset
from rs_tools._src.preprocessing.normalize import apply_spectral_normalizer, apply_coordinate_normalizer
import xarray as xr
from torch.utils.data import Dataset, DataLoader, RandomSampler
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.datamodule.utils import get_split

# TODO: Import from ITI rather than rs_tools
from rs_tools._src.datamodule.base import BaseDataset

class GeoDataset(BaseDataset):
    def __init__(
        self,
        data_dir: List[str],
        editors: List[Editor],
        splits_dict: Dict=None,
        ext: str="nc",
        limit: int=None,
        bands: List[int]=None,
        include_coords: bool=True,
        include_cloudmask: bool=True, 
        include_nanmask: bool=True,
        band_norm: xr.Dataset=None,
        coord_norm: bool=False,
        **kwargs
    ):
        self.data_dir = data_dir
        self.editors = editors
        self.splits_dict = splits_dict
        self.ext = ext
        self.limit = limit
        self.bands = bands
        self.include_coords = include_coords
        self.include_cloudmask = include_cloudmask
        self.include_nanmask = include_nanmask
        self.band_norm = band_norm
        self.coord_norm = coord_norm

        self.data = self.get_files()

        super().__init__(
            data=self.data,
            editors=self.editors,
            ext=self.ext,
            limit=self.limit,
            **kwargs
        )

    def get_files(self):
        # Get filenames from data_dir
        self.filenames = get_list_filenames(data_path=self.data_dir, ext=self.ext)
        # split files based on split criteria
        files = get_split(filenames=self.filenames, split_dict=self.splits_dict)
        return files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load dataset
        ds: xr.Dataset = xr.load_dataset(self.file_list[idx], engine="netcdf4")

        # apply normalization per entity
        if self.band_norm is not None:
            radiances: np.ndarray = apply_spectral_normalizer(ds.values, self.band_norm)
        else:
            radiances: np.ndarray = ds.values
        
        if self.coord_norm:
            # TODO: Check if this is extracting x/y or lat/lon
            coords_: np.ndarray = apply_coordinate_normalizer(ds.spatial_coords)
        else:
            coords_: np.ndarray = ds.spatial_coords

        # TODO: Add band selection
        # TODO: Add checks & cloudmask selection
        # TODO: Add checks & nanmask creation
        # TODO: Apply transforms

        

        #==========================
        # - Load it from file
        # - Reshape Dataset to Array
        # - choose the channels (bands, coordinates, cloud-mask, etc)
        # - Convert to numpy/etc
        # - apply transforms
        # - output dictionary
        # Psuedo-Code
        # # Load dataset
        # ds: xr.Dataset = xr.load_dataset(...)
        # # apply normalization per entity
        # radiances: Array["C H W"] = apply_spectral_normalizer(ds.values)
        # coords_: Array["2 H W"] = apply_coordinate_normalizer(ds.spatial_coords)
        # time_: Array["T H W"] = apply_time_normalizer(ds.time)
        # mask_: Array["M H W"] = apply_mask_normalizer(ds.mask)
        # MASKS
        # Q: per band, per space
        # nan_mask: Array["H W"] = np.isnan()
        # pass