from __future__ import annotations

import gc
import os
import autoroot
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
import multiprocessing
import xarray as xr
from loguru import logger
from rs_tools._src.utils.io import get_list_filenames
from tqdm import tqdm

def _check_filetype(file_type: str) -> bool:
    """checks instrument for GOES data."""
    if file_type in ["nc", "npy", "npz"]:
        return True
    else:
        msg = "Unrecognized file type"
        msg += f"\nNeeds to be 'nc', 'npy' or 'npz'. Others are not yet tested"
        raise ValueError(msg)

def create_fov_mask(shape, fov_radius):
    """
    Function to create mask for specified field of view.
    """
    # Create coordinate grids
    y, x = np.ogrid[:shape[0], :shape[1]]
    # Calculate center points
    center_y, center_x = shape[0] // 2, shape[1] // 2
    # Calculate distance from center for each point
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Normalize distances by max possible distance (corner to center)
    max_dist = np.sqrt((center_x)**2 + (center_y)**2)
    normalized_dist = dist_from_center / max_dist
    # Create mask for specified field of view
    mask = normalized_dist <= fov_radius
    return mask

class CenterWeightedCrop():
    def __init__(self, patch_shape, fov_radius=0.6):
        self.patch_shape = patch_shape
        self.fov_radius = fov_radius
        self.max_attempts = 5
    def __call__(self, ds):
        assert ds['x'].shape[0] >= self.patch_shape[0], 'Invalid dataset shape: %s' % str(dataset[self.x].shape)
        assert ds['y'].shape[0] >= self.patch_shape[1], 'Invalid dataset shape: %s' % str(dataset[self.y].shape)

        # get x/y grid
        x_grid, y_grid = np.meshgrid(np.arange(0, ds.x.shape[0], 1), np.arange(0, ds.y.shape[0], 1))

        # create mask for valid coordinates within desired field of view
        # NOTE: This masks from the center to the image edge, rather than disk edge
        valid_mask = create_fov_mask(shape=(ds.x.shape[0], ds.y.shape[0]), fov_radius=self.fov_radius)

        # get coordinate pairs for valid points
        coords_on_disk = np.column_stack((x_grid[valid_mask], y_grid[valid_mask]))
        del x_grid, y_grid

        # pick random x/y index
        attempts = 0
        while attempts <= self.max_attempts:
            random_idx = np.random.randint(0, len(coords_on_disk))
            x, y = tuple(coords_on_disk[random_idx])
            # define patch boundaries
            xmin = x - self.patch_shape[0] // 2
            ymin = y - self.patch_shape[1] // 2
            xmax = x + self.patch_shape[0] // 2
            ymax = y + self.patch_shape[1] // 2

            # crop patch
            patch_ds = ds.sel({'x': slice(ds['x'][xmin], ds['x'][xmax - 1]),
                                'y': slice(ds['y'][ymin], ds['y'][ymax - 1])})
            # check that there are no constant channels
            if not np.any(np.nanstd(ds.Rad.values, axis=(1, 2)) == 0):
                return patch_ds, xmin, ymin
            attempts += 1
        logger.info(f'Could not find valid patch after {self.max_attempts} cropping attempts')
        return patch_ds, xmin, ymin

class Patcher():
    """
    A class for preprocessing and saving patches from NetCDF files.
    Patches are weighted towards the center of the image.

    Attributes:
        read_path (str): The path to the directory containing the NetCDF files.
        save_path (str): The path to save the patches.
        patch_size (int): The size of each patch.
        num_patches (int): The number of patches to extract.
        save_filetype (str): The file type to save patches as. Options are [nc, np, tif].

    Methods:
        nc_files(self) -> List[str]: Returns a list of all NetCDF filenames in the read_path directory.
        save_patches(self): Preprocesses and saves patches from the NetCDF files.
    """
    def __init__(
        self,
        read_path: str,
        save_path: str,
        patch_size: int,
        num_patches: int,
        save_filetype: str,
    ):
        self.read_path = read_path
        self.save_path = save_path
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.save_filetype = save_filetype

        self.crop = CenterWeightedCrop(patch_shape=(patch_size, patch_size))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def nc_files(self) -> list[str]:
        """
        Returns a list of all NetCDF filenames in the read_path directory.

        Returns:
            List[str]: A list of NetCDF filenames.
        """
        # get list of all filenames within the path
        files = get_list_filenames(self.read_path, ".nc")
        return files

    def stack(self, ds):
        data = ds.Rad.values
        lats = ds.latitude.values
        lats = np.expand_dims(lats, axis=0)
        lons = ds.longitude.values
        lons = np.expand_dims(lons, axis=0)
        cloud_mask = ds.cloud_mask.values
        cloud_mask = np.expand_dims(cloud_mask, axis=0)

        stack = np.concatenate([data, cloud_mask, lats, lons], axis=0)
        stack = stack.astype('float32')
        del data, lats, lons, cloud_mask
        return stack

    def save_patch(
        self, 
        patch_ds: xr.Dataset, 
        save_path: str, 
        filename: str, 
        patch_num: int, 
        xmin: int, 
        ymin: int):
        """
        Saves the patch as a NetCDF file.

        Args:
            patch_ds (xr.Dataset): The patch dataset.
            save_path (str): The path to save the patch.
            filename (str): The filename of the patch.
            patch_num (int): The patch number.
            xmin (int): The minimum x-coordinate of the patch.
            ymin (int): The minimum y-coordinate of the patch
        """
        save_name = f"{filename}_patch_{patch_num}_{xmin}-{ymin}.{self.save_filetype}"
        # save patch
        if self.save_filetype == "nc":
            patch_ds.to_netcdf(os.path.join(save_path, save_name))
        elif self.save_filetype == "npy":
            stack = self.stack(patch_ds)
            np.save(os.path.join(save_path, save_name), stack)
        elif self.save_filetype == 'npz':
            stack = self.stack(patch_ds)
            np.savez(os.path.join(save_path, save_name), stack)


    def patch(self):
        """
        Preprocesses and saves patches from the NetCDF files.
        """
        pbar = tqdm(self.nc_files())

        for ifile in pbar:
            # extract & log timestamp
            itime = str(Path(ifile).name).split("_")[0]
            pbar.set_description(f"Processing: {itime}")
            # open dataset
            ds: xr.Dataset = xr.load_dataset(ifile, engine="netcdf4")
            for i in tqdm(range(self.num_patches)):
                # crop patch
                patch_ds, xmin, ymin = self.crop(ds)
                # save patch
                self.save_patch(
                    patch_ds = patch_ds,
                    save_path = self.save_path,
                    filename = str(Path(ifile).name).split(".")[0],
                    patch_num = i,
                    xmin = xmin,
                    ymin = ymin
                )
                # clear memory
                del patch_ds
                gc.collect()
        return True

def prepatch(
    patch_size: int = 256,
    num_patches: int = 100,
    read_path: str = "./",
    save_path: str = "./",
    save_filetype: str = "nc",
):
    """
    Patches satellite data into smaller patches for training.
    Args:
        read_path (str, optional): The path to read the input files from. Defaults to "./".
        save_path (str, optional): The path to save the extracted patches. Defaults to "./".
        patch_size (int, optional): The size of each patch. Defaults to 256.
        num_patches (int, optional): The number of patches to extract. Defaults to 100.
        save_filetype (str, optional): The file type to save patches as. Options are [nc, np]

    Returns:
        None
    """
    _check_filetype(file_type=save_filetype)

    # Initialize Prepatcher
    logger.info(f"Patching Files...: {read_path}")
    logger.info(f"Initializing CenterPatcher...")
    patcher = Patcher(
        read_path=read_path,
        save_path=save_path,
        patch_size=patch_size,
        num_patches=num_patches,
        save_filetype=save_filetype,
    )
    # Extract files
    files = patcher.nc_files()
    # Define number of CPUs
    cpus = int(multiprocessing.cpu_count() // 2)
    # Split files into chunks
    chunk_files = np.array_split(files, cpus)
    chunk_list = [list(chunk) for chunk in chunk_files]

    logger.info(f"Patching {len(chunk_files)} splits using {cpus} CPUs...")
    # Patch files
    with multiprocessing.Pool(cpus) as p:
        p.map(patcher.patch(), chunk_list)
        p.close()
        p.join()

    logger.info(f"Finished Patching Script...!")

if __name__ == "__main__":
    typer.run(prepatch)
