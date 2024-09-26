from __future__ import annotations

import gc
import os
import autoroot
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
import xarray as xr
from loguru import logger
from satpy import Scene
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.geoprocessing.msg.reproject import add_msg_crs
from tqdm import tqdm
from xrpatcher._src.base import XRDAPatcher


def _check_filetype(file_type: str) -> bool:
    """checks filetype."""
    if file_type in ["np", "npz", "tif"]:
        return True
    else:
        msg = "Unrecognized file type"
        msg += f"\nNeeds to be 'np', 'npz' or 'tif'. Others are not yet tested"
        raise ValueError(msg)


def _check_nan_count(arr: np.array, nan_cutoff: float) -> bool:
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

    pct_nan = nan_count / total_count

    if pct_nan <= nan_cutoff:
        return True
    else:
        return False


@dataclass(frozen=True)
class MSGRawPatcher:
    """
    A class for preprocessing and saving patches from raw MSG files.

    Attributes:
        read_path (str): The path to the directory containing the MSG files.
        save_path (str): The path to save the patches.
        patch_size (int): The size of each patch.
        stride_size (int): The stride size for generating patches.
        nan_cutoff (float): The cutoff value for allowed NaN count in a patch.
        save_filetype (str): The file type to save patches as. Options are [np, npz, tif].
        stack_coords (bool): Whether to stack latitude and longitude data into a single DataArray.

    Methods:
        nc_files(self) -> List[str]: Returns a list of all NetCDF filenames in the read_path directory.
        save_patches(self): Preprocesses and saves patches from the NetCDF files.
    """

    read_path: str
    save_path: str
    patch_size: int
    stride_size: int
    nan_cutoff: float
    save_filetype: str
    stack_coords: bool = True

    @property
    def nat_files(self) -> list[str]:
        """
        Returns a list of all native filenames in the read_path directory.

        Returns:
            List[str]: A list of native filenames.
        """
        # get list of all filenames within the path
        files = get_list_filenames(self.read_path, ".nat")
        return files

    def load_file(self, filename: str) -> xr.Dataset:
        """Function to load native MSG file using satpy.

        Args:
            filename (str): Filename of the MSG file to load.

        Returns:
            xr.Dataset: Dataset containing the MSG data.
        """
        scn = Scene(reader="seviri_l1b_native", filenames=[filename])
        datasets = scn.available_dataset_names()
        scn.load(datasets[1:], generate=False, calibration='radiance')
        ds = scn.to_xarray()
        return ds

    def stack_data(self, ds: xr.Dataset) -> xr.DataArray:
        """
        Stacks band, latitude, and longitude data into a single DataArray.

        Args:
            ds (xr.Dataset): Raw MSG dataset.

        Returns:
            xr.DataArray: Data array.
        """

        variables = ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

        # stack all variables into one array:
        stacked_data = np.stack([ds[var].values for var in variables], axis=0)
        if self.stack_coords:
            stacked_data = np.concatenate([stacked_data, ds['latitude'].values[np.newaxis, :, :]], axis=0)
            stacked_data = np.concatenate([stacked_data, ds['longitude'].values[np.newaxis, :, :]], axis=0)
            variables.append('latitude')
            variables.append('longitude')

        stacked_dataarray = xr.DataArray(stacked_data, dims=['band', 'y', 'x'], coords={'band': variables})
        return stacked_dataarray

    def save_patches(self):
        """
        Preprocesses and saves patches from the NetCDF files.
        """
        pbar = tqdm(self.nat_files)

        for ifile in pbar:
            # extract & log timestamp
            itime = str(Path(ifile).name).split(".")[0].split("-")[-1]
            pbar.set_description(f"Processing: {itime}")
            # open dataset
            ds = self.load_file(ifile)
            if self.save_filetype == "tif":
                # add CRS
                ds = add_msg_crs(ds)
            # stack data
            da = self.stack_data(ds)

            band_names = da.band.values
                
            # define patch parameters
            patches = dict(x=self.patch_size, y=self.patch_size)
            strides = dict(x=self.stride_size, y=self.stride_size)
            # start patching
            patcher = XRDAPatcher(da=da, patches=patches, strides=strides)

            # check if save path exists, and create if not
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            for i, ipatch in tqdm(enumerate(patcher), total=len(patcher)):
                data = ipatch.data  # extract data
                # logger.info(f'stride size {self.stride_size} ')
                if _check_nan_count(data, self.nan_cutoff):
                    if self.save_filetype == "tif":
                        # reconvert to dataset to attach band_wavelength and time
                        # ds.attrs['band_names'] = [str(i) for i in ds.band.values]
                        # compile filename
                        file_path = Path(self.save_path).joinpath(
                            f"{itime}_patch_{i}.tif"
                        )
                        # remove file if it already exists
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        # add band names as attribute
                        ipatch.attrs["band_names"] = band_names
                        # save patch to tiff
                        ipatch.rio.to_raster(file_path)
                    elif self.save_filetype == "np":
                        # save as numpy files
                        np.save(
                            Path(self.save_path).joinpath(
                                f"{itime}_patch_{i}"
                            ),
                            data,
                        )
                    elif self.save_filetype == "npz":
                        # save as numpy files
                        np.savez_compressed(
                            Path(self.save_path).joinpath(
                                f"{itime}_patch_{i}"
                            ),
                            data,
                        )
                else:
                    pass
                    # logger.info(f'NaN count exceeded for patch {i} of timestamp {itime}.')


def prepatch(
    read_path: str = "/home/anna.jungbluth/data/L1b/",
    save_path: str = "/home/anna.jungbluth/data/patches/",
    patch_size: int = 256,
    stride_size: int = 192,
    nan_cutoff: float = 0.5,
    save_filetype: str = "npz",
):
    """
    Patches satellite data into smaller patches for training.
    Args:
        read_path (str, optional): The path to read the input files from. Defaults to "./".
        save_path (str, optional): The path to save the extracted patches. Defaults to "./".
        patch_size (int, optional): The size of each patch. Defaults to 256.
        stride_size (int, optional): The stride size for patch extraction. Defaults to 256.
        nan_cutoff (float): The cutoff value for allowed NaN count in a patch. Defaults to 0.1.
        save_filetype (str, optional): The file type to save patches as. Options are [nc, np]

    Returns:
        None
    """
    _check_filetype(file_type=save_filetype)

    # Initialize Prepatcher
    logger.info(f"Patching Files...: {read_path}")
    logger.info(f"Initializing Prepatcher...")
    prepatcher = MSGRawPatcher(
        read_path=read_path,
        save_path=save_path,
        patch_size=patch_size,
        stride_size=stride_size,
        nan_cutoff=nan_cutoff,
        save_filetype=save_filetype,
    )
    logger.info(f"Patching Files...: {save_path}")
    prepatcher.save_patches()

    logger.info(f"Finished Prepatching Script...!")


if __name__ == "__main__":
    """
    python scripts/pipeline/prepatch.py --read-path "/path/to/netcdf/file" --save-path /path/to/save/patches
    """
    typer.run(prepatch)
