from xrpatcher._src.base import XRDAPatcher
import xarray as xr
import numpy as np
from rs_tools._src.geoprocessing.crs import add_crs_from_rio
import pandas as pd
from datetime import datetime
from rs_tools._src.preprocessing.nans import check_nan_count
from tqdm import tqdm


def modis_patch_to_file(
        file_name: str="./",
        variable: str="",
        patch_size: int=128,
        stride_size: int=128,
        nan_cutoff: float=0.1,
        fill_value: float=0.0,
        save_dir: str="./",
        overwrite: bool=True,
        save_file_type: str="npy",
):
    
    if not save_dir.is_dir():
        save_dir.mkdir(parents=True, exist_ok=True)


    # load dataset
    ds = xr.open_dataset(file_name, engine="netcdf4")

    # add crs to dataset
    ds = add_crs_from_rio(ds)

    # grab time components
    time_stamp = pd.to_datetime(ds.time.values.squeeze())
    itime_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")

    # add cloud mask (if applicable)
    if "cloud_mask" in list(ds.variables.keys()):
        ds = ds.assign_coords({"cloud_mask": ds.cloud_mask})
        ds["cloud_mask"].attrs = {}

    # select variable
    da = ds[variable]

    # select appropriate dimension names
    patches = {da.rio.x_dim: patch_size, da.rio.y_dim: patch_size}
    strides = {da.rio.x_dim: stride_size, da.rio.y_dim: stride_size}

    # initialize patcher
    patcher = XRDAPatcher(da=da, patches=patches, strides=strides)

    pbar = tqdm(enumerate(patcher), total=len(patcher), leave=False)

    for i, ipatch in pbar:

        # check nan counts
        # fill values - zeros
        ipatch = ipatch.where(ipatch != 0.0, np.nan)
        # fill values - negative numbers
        ipatch = ipatch.where(ipatch >= 0.0, np.nan)

        # check nan counts
        if check_nan_count(ipatch.data, nan_cutoff):

            # fill nans with value
            ipatch = ipatch.where(ipatch != np.nan, fill_value)

            ipatch = ipatch.to_dataset(name=variable)

            # create file path
            
            save_file_name = f"{itime_name}_patch_{i}"
            
            if save_file_type in ["netcdf", "netcdf4"]:
                _save_to_netcdf(ipatch, save_dir=save_dir, save_file_name=save_file_name, overwrite=overwrite)
            elif save_file_type in ["zarr"]:
                _save_to_zarr(ipatch, save_dir=save_dir, save_file_name=save_file_name, overwrite=overwrite)
            elif save_file_type in ["tiff", "tif", "geotiff"]:
                _save_to_tiff(ipatch[variable], save_dir=save_dir, save_file_name=save_file_name, overwrite=overwrite, fill_value=fill_value)
            elif save_file_type in ["npy", "numpy"]:
                _save_to_numpy(ipatch[variable], save_dir=save_dir, save_file_name=save_file_name, overwrite=overwrite, fill_value=fill_value)
            else:
                raise ValueError(f"Unrecognized filetype: {save_file_type}")

        else:
            continue

    return None


def _save_to_numpy(
        patch: xr.DataArray,
        save_file_name: str,
        save_dir: str,
        overwrite: bool=False,
        fill_value: float=0.0
):
        
        # save as numpy files
        np.save(save_dir.joinpath(f"{save_file_name}_{patch.name}"), patch.values)
        np.save(save_dir.joinpath(f"{save_file_name}_cloud_mask"), patch.cloud_mask.values)

        # just in case we have curvilinear coordinates
        if patch.longitude.ndim == 1:
            np.save(save_dir.joinpath(f"{save_file_name}_latitude"), patch.latitude.values)
            np.save(save_dir.joinpath(f"{save_file_name}_longitude"), patch.longitude.values)
        else:
            X, Y = np.meshgrid(patch.latitude.values, patch.longitude.values, indexing="ik")
            np.save(save_dir.joinpath(f"{save_file_name}_latitude"), X)
            np.save(save_dir.joinpath(f"{save_file_name}_longitude"), Y)
        

def _save_to_tiff(
        patch: xr.DataArray,
        save_file_name: str,
        save_dir: str,
        overwrite: bool=False,
        fill_value: float=0.0
):  
        ifile_path = save_dir.joinpath(f"{save_file_name}.tiff")

        if overwrite and ifile_path.is_file():
            # remove file if it already exists
            ifile_path.unlink()

        # create a dataset
        patch = patch.to_dataset(dim="band")

        # remove attributes (otherwise it doesnt save properly...)
        patch.attrs = {}

        try:
            patch = patch.reset_coords(names='cloud_mask', drop=False)
            patch = patch.where(patch != np.nan, 4.0)
        except:
            pass
        
        # save with rasterio
        patch.rio.to_raster(ifile_path)

        return None


def _save_to_zarr(
        patch: xr.Dataset,
        save_file_name: str,
        save_dir: str,
        overwrite: bool=False,
):  
    
    ifile_path = save_dir.joinpath(f"{save_file_name}.zarr")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()

    patch.to_zarr(ifile_path, mode="w")
    return None


def _save_to_netcdf(
        patch: xr.Dataset,
        save_file_name: str,
        save_dir: str,
        overwrite: bool=False
):  
    
    ifile_path = save_dir.joinpath(f"{save_file_name}.nc")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()

    patch.to_netcdf(ifile_path, engine="netcdf4")
    return None

