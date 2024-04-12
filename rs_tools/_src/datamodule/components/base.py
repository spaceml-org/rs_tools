from __future__ import annotations

import numpy as np
from typing import List
from loguru import logger
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset
import xarray as xr

class BaseDataset(Dataset):
    def __init__(
        self,
        file_list,
        transforms=None,
        bands: List[int]=None,
        coords: bool=False,
        masks: List[int]=None,
        include_time: bool=True
    ):
        # TODO: Check that files exist
        # TODO: Check that files are netcdf
        # TODO: Make sure it comes out as dict

        self.file_list = file_list
        self.transforms = transforms

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Loading NetCDF
        # - Load it from file
        # - Some checks...?
        # - Return it!
        return xr.open_dataset(idx, engine="netcdf4")
        


        #
        #
        #
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