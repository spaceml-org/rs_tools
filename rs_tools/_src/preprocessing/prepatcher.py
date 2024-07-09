from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
import xarray as xr
from loguru import logger
from rs_tools._src.utils.io import get_list_filenames
from tqdm import tqdm
from xrpatcher._src.base import XRDAPatcher


def _check_filetype(file_type: str) -> bool:
    """checks instrument for GOES data."""
    if file_type in ["nc", "np", "tif"]:
        return True
    else:
        msg = "Unrecognized file type"
        msg += f"\nNeeds to be 'nc', 'np', or 'tif'. Others are not yet tested"
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
class PrePatcher:
    """
    A class for preprocessing and saving patches from NetCDF files.

    Attributes:
        read_path (str): The path to the directory containing the NetCDF files.
        save_path (str): The path to save the patches.
        patch_size (int): The size of each patch.
        stride_size (int): The stride size for generating patches.
        nan_cutoff (float): The cutoff value for allowed NaN count in a patch.
        save_filetype (str): The file type to save patches as. Options are [nc, np, tif].

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

    @property
    def nc_files(self) -> list[str]:
        """
        Returns a list of all NetCDF filenames in the read_path directory.

        Returns:
            List[str]: A list of NetCDF filenames.
        """
        # get list of all filenames within the path
        files = get_list_filenames(self.read_path, ".nc")
        return files

    def save_patches(self):
        """
        Preprocesses and saves patches from the NetCDF files.
        """
        pbar = tqdm(self.nc_files)

        for ifile in pbar:
            # extract & log timestamp
            itime = str(Path(ifile).name).split("_")[0]
            pbar.set_description(f"Processing: {itime}")
            # open dataset
            ds = xr.open_dataset(ifile, engine="netcdf4")

            if self.save_filetype == "tif":
                # concatenate variables
                ds_temp = xr.concat(
                    [ds.cloud_mask, ds.latitude, ds.longitude], dim="band"
                )
                # name data variables "Rad"
                ds_temp = ds_temp.to_dataset(name="Rad")
                ds_temp = ds_temp.drop(["cloud_mask", "latitude", "longitude"])
                ds_temp = ds_temp.assign_coords(
                    band=["cloud_mask", "latitude", "longitude"]
                )
                # merge with original dataset
                ds = xr.merge([ds_temp.Rad, ds.Rad])
                # store band names to be attached to da later
                band_names = [str(i) for i in ds.band.values]
                del ds_temp
                gc.collect()
                
            # extract radiance data array
            da = ds.Rad
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
                    if self.save_filetype == "nc":
                        # reconvert to dataset to attach band_wavelength and time
                        ipatch = ipatch.to_dataset(name="Rad")
                        ipatch = ipatch.assign_coords({"time": ds.time.values})
                        ipatch = ipatch.assign_coords(
                            {"band_wavelength": ds.band_wavelength.values}
                        )
                        # compile filename
                        file_path = Path(self.save_path).joinpath(
                            f"{itime}_patch_{i}.nc"
                        )
                        # remove file if it already exists
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        # save patch to netcdf
                        ipatch.to_netcdf(
                            Path(self.save_path).joinpath(f"{itime}_patch_{i}.nc"),
                            engine="netcdf4",
                        )
                    elif self.save_filetype == "tif":
                        # reconvert to dataset to attach band_wavelength and time
                        # ds.attrs['band_names'] = [str(i) for i in ds.band.values]
                        # compile filename
                        file_path = Path(self.save_path).joinpath(
                            f"{itime}_patch_{i}.nc"
                        )
                        # remove file if it already exists
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        # add band names as attribute
                        ipatch.attrs["band_names"] = band_names
                        # save patch to tiff
                        ipatch.rio.to_raster(
                            Path(self.save_path).joinpath(f"{itime}_patch_{i}.tif")
                        )
                    elif self.save_filetype == "np":
                        # save as numpy files
                        np.save(
                            Path(self.save_path).joinpath(
                                f"{itime}_radiance_patch_{i}"
                            ),
                            data,
                        )
                        np.save(
                            Path(self.save_path).joinpath(
                                f"{itime}_latitude_patch_{i}"
                            ),
                            ipatch.latitude.values,
                        )
                        np.save(
                            Path(self.save_path).joinpath(
                                f"{itime}_longitude_patch_{i}"
                            ),
                            ipatch.longitude.values,
                        )
                        np.save(
                            Path(self.save_path).joinpath(
                                f"{itime}_cloudmask_patch_{i}"
                            ),
                            ipatch.cloud_mask.values,
                        )
                else:
                    pass
                    # logger.info(f'NaN count exceeded for patch {i} of timestamp {itime}.')


def prepatch(
    read_path: str = "./",
    save_path: str = "./",
    patch_size: int = 256,
    stride_size: int = 256,
    nan_cutoff: float = 0.5,
    save_filetype: str = "nc",
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
    prepatcher = PrePatcher(
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
