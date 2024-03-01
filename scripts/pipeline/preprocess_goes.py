import autoroot
import numpy as np
import rioxarray
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
from tqdm import tqdm
from rs_tools._src.utils.io import get_list_filenames
import typer
from loguru import logger
import xarray as xr
import datetime
from rs_tools._src.geoprocessing.goes.interp import resample_rioxarray
from rs_tools._src.geoprocessing.goes.validation import correct_goes16_bands, correct_goes16_satheight
from rs_tools._src.geoprocessing.goes.reproject import add_goes16_crs
from rs_tools._src.geoprocessing.reproject import convert_lat_lon_to_x_y
import pandas as pd
from datetime import datetime


def parse_goes16_dates_from_file(file: str):
    timestamp = Path(file).name.replace("-","_").split("_")
    return datetime.strptime(timestamp[-2][1:], "%Y%j%H%M%S%f").strftime("%Y%j%H%M")


@dataclass
class GOES16GeoProcessing:
    resolution: float = 1_000 # [m]
    read_path: str = "./"
    save_path: str = "./"
    region: Tuple[str] = (-130, -15, -90, 5)
    resample_method: str = "bilinear"

    @property
    def goes_files(self) -> List[str]:
        # get a list of all filenames within the path
        # get all modis files
        files = get_list_filenames(self.read_path, ".nc")
        return files
    
    def preprocess_fn(self, ds: xr.Dataset) -> xr.Dataset:

        ds = ds.copy()

        ds = correct_goes16_satheight(ds)
        ds = correct_goes16_bands(ds)
        ds = add_goes16_crs(ds)
        
        # subset data
        lon_bnds = (self.region[0], self.region[2])
        lat_bnds = (self.region[1], self.region[3])
        x_bnds, y_bnds = convert_lat_lon_to_x_y(ds.FOV.crs, lon=lon_bnds, lat=lat_bnds, )
        ds = ds.sortby("x").sortby("y")
        ds = ds.sel(y=slice(y_bnds[0], y_bnds[1]),x=slice(x_bnds[0], x_bnds[1]))
        
        # resampling
        ds_subset = resample_rioxarray(ds, resolution=self.resolution, method=self.resample_method)

        # assign coordinates
        ds_subset = ds_subset[["Rad", "DQF"]]
        time_stamp = pd.to_datetime(ds.t.values)
        time_stamp = time_stamp.strftime("%Y-%m-%d %H:%M")
        ds_subset[["Rad", "DQF"]] = ds_subset[["Rad", "DQF"]].expand_dims({"band":ds.band.values, "time":[time_stamp]})
        ds_subset = ds_subset.drop_vars(["t", "y_image", "x_image", "goes_imager_projection"])
        # assign bands
        ds_subset.band.attrs = ds.band.attrs
        ds_subset = ds_subset.assign_coords({"band_wavelength": ds.band_wavelength.values})
        ds_subset.band_wavelength.attrs = ds.band_wavelength.attrs
        
        return ds_subset

    def preprocess_files(self):

        # get unique times
        unique_times = list(set(map(parse_goes16_dates_from_file, self.goes_files)))

        pbar_time = tqdm(unique_times)

        for itime in pbar_time:

            pbar_time.set_description(f"Processing: {itime}")

            # get files from unique times
            files = list(filter(lambda x: unique_times[0] in x, self.goes_files))

            # open
            ds = [xr.open_mfdataset(ifile, preprocess=self.preprocess_fn, concat_dim="band", combine="nested") for ifile in files]
            # reinterpolate to match coordinates of first image
            ds = [ds[0]] + [ids.interp(x=ds[0].x, y=ds[0].y) for ids in ds[1:]]
            # concatentate
            ds = xr.concat(ds, dim="band")
            
            # TODO: Keep relevant attributes...
            attrs_rad = ds["Rad"].attrs

            ds["Rad"].attrs = {}
            ds["Rad"].attrs = dict(
                long_name=attrs_rad["long_name"],
                standard_name=attrs_rad["standard_name"],
                units=attrs_rad["units"],
            )
            ds["DQF"].attrs = {}

            ds.to_netcdf(Path(self.save_path).joinpath(f"{itime}_goes16.nc"), engine="netcdf4")



def preprocess_goes16(
        resolution: float = 1000,
        read_path: str = "./",
        save_path: str = "./"
):
    """
    Downloads MODIS TERRA and GOES 16 files for the specified period, region, and save path.

    Args:
        period (List[str], optional): The period of time to download files for. Defaults to ["2020-10-01", "2020-10-31"].
        region (Tuple[str], optional): The geographic region to download files for. Defaults to (-180, -90, 180, 90).
        save_path (str, optional): The path to save the downloaded files. Defaults to "./".

    Returns:
        None
    """
    logger.info(f"Starting Script...")

    logger.info(f"Initializing GeoProcessor...")
    modis_geoprocessor = GOES16GeoProcessing(
        resolution=resolution, read_path=read_path, save_path=save_path
        )
    logger.info(f"Saving Files...")
    modis_geoprocessor.preprocess_files()

    logger.info(f"Finished Script...!")




if __name__ == '__main__':
    """
    python scripts/pipeline/preprocess_modis.py --read-path "/home/juanjohn/data/rs/modis/raw" --save-path /home/juanjohn/data/rs/modis/analysis
    """
    typer.run(preprocess_goes16)
