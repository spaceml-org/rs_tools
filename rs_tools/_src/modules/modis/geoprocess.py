from dataclasses import (
    dataclass,
    field,
)
from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    List,
)

import geopandas as gpd
from georeader.griddata import footprint as swath_footprint
from odc.geo.crs import CRS
from odc.geo.geom import (
    BoundingBox,
    Geometry,
)
import pandas as pd
from rasterio.errors import RasterioIOError
from tqdm import tqdm
import xarray as xr

from rs_tools._src.modules.modis.ops import load_modis_bands_raw, load_modis_cloud_mask_raw
from rs_tools._src.data.modis.bands import CALIBRATION_CHANNELS
from rs_tools._src.utils.io import get_list_filenames
from rs_tools._src.data.modis import MODIS_ID_TO_NAME


@dataclass
class RawMODIS:
    """
    Represents a dataset of raw MODIS files.

    Attributes:
        file_dir (str): The directory where the files are located.
        file_ext (str): The file extension of the MODIS files (default: ".hdf").
        calibration (str): The calibration type (default: "radiance").
        satellite_name (str): The name of the satellite.
        satellite_id (str): The ID of the satellite.
        filename_regex (str): The regular expression pattern to match the file names.
        transforms (Callable[[Any], Any] | None): A function to apply transformations to the dataset (default: None).
        fill_value (float): The fill value for missing data (default: 0.0).
    """
    file_dir: str
    file_ext: str = ".hdf"
    calibration: str = "radiance"
    satellite_name: str = ""
    satellite_id: str = ""
    filename_regex: str = None
    transforms: Callable[[Any], Any] | None = None
    fill_value: float = 0.0

    def __post_init__(self):
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        # get all filenames
        files = get_list_filenames(self.file_dir, self.file_ext)

        crs = CRS("WGS84")

        # loop through files
        geo_dataframes = []

        pbar = tqdm(files, leave=False)
        for ifile in pbar:

            try:
                # get the name

                ifile_name = Path(ifile).name
                pbar.set_description(f"FileName: {ifile_name}")

                # match with regex
                match = re.match(filename_regex, str(ifile_name))

                if not match:
                    continue

                year = match.group("year")
                doy = match.group("day_of_year")
                time = match.group("time")
                time_str = f"{year}{doy}{time}"
                timestamp = pd.to_datetime(time_str, format="%Y%j%H%M")

                # open raw file
                ds = self.load_raw_files(ifile)

                # calculate footprint
                polygon = swath_footprint(ds.longitude.values, ds.latitude.values)
                # get polygon
                # assign CRS
                # create pandas geodateframe
                df = pd.DataFrame(
                    {
                        "time": timestamp,
                        "satellite_id": self.satellite_id,
                        "satellite_name": self.satellite_name,
                        "full_path": str(ifile),
                    },
                    index=[0],
                )

                # convert the list of polygons into a GeoSeries
                geometry = gpd.GeoSeries(polygon)

                igdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

                # create a GeoDataFrame
                geo_dataframes.append(igdf)

            except RasterioIOError:
                continue

        # concatenate dateframes
        geo_dataframes = pd.concat(geo_dataframes, ignore_index=True)

        # clean and save the dataframe
        geo_dataframes = geo_dataframes.drop_duplicates().reset_index(drop=True)

        # save geodataframes
        self.geo_dataframes = geo_dataframes

        self.src_crs = crs
        self.src_res = (1_000, 1_000)
        try:
            self.bands = CALIBRATION_CHANNELS[self.calibration]
        except KeyError:
            pass
            
    def __str__(self):
        """
        Returns a string representation of the dataset.
        """
        return (
            "GOES16 RAW Dataset"
            + f"\nCRS: {self.src_crs}"
            + f"\nResolution: {self.src_res}"
            + f"\nNumber Files: {len(self)}"
            + f"\nNumber TimeStamps: {len(self.time_stamps)}"
            + f"\nBands: {self.bands}"
            + f"\nTransforms: {self.transforms}"
            + f"\nFile Dir: {self.file_dir}"
            + f"\nFile Ext: {self.file_ext}"
        )

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.geo_dataframes)

    @property
    def time_stamps(self):
        """
        Returns a list of unique timestamps in the dataset.
        """
        return list(self.geo_dataframes["time"].unique())

    @property
    def total_indices(self):
        """
        Returns the total number of indices in the dataset.
        """
        return len(self.geo_dataframes)
    
    @abstractmethod
    def load_raw_files(self, ifile: str):
        pass

    def iloc(self, idx) -> xr.Dataset:
        """
        Retrieves a single band from the dataset based on the given index.

        Parameters:
            idx (int): The index of the band to retrieve.

        Returns:
            ndarray: The single band data.
        """
        full_path = self.geo_dataframes.iloc[idx]["full_path"]
        return self.load_raw_files(full_path)

    def __getitem__(self, idx):
        """
        Retrieves a single band from the dataset based on the given index.

        Parameters:
            idx (int): The index of the band to retrieve.

        Returns:
            ndarray: The single band data.
        """
        # filter dataframe for files
        xdset = self.iloc(idx)

        if self.transforms:
            xdset = self.transforms(xdset)

        return xdset


@dataclass
class RawTerra(RawMODIS):
    """
    Represents a raw MODIS Terra dataset.

    Attributes:
        filename_regex (str): Regular expression pattern for matching the filename of the dataset.
        satellite (str): Name of the satellite (Terra).
        satellite_id (str): Identifier for the satellite (MOD021KM).
    """

    filename_regex: str = (
        r"^MOD021KM\.A(?P<year>\d{4})(?P<day_of_year>\d{3})\.(?P<time>\d{4})\.(?P<version>\d+)\.(?P<timestamp>\d+)\.hdf"
    )
    satellite: str = "terra"
    satellite_id: str = "MOD021KM"
    
    def load_raw_files(self, ifile):
        """
        Load raw MODIS bands from the given file.

        Args:
            ifile (str): Path to the input file.

        Returns:
            object: Loaded MODIS bands data.
        """
        return load_modis_bands_raw(ifile, calibration=self.calibration)


@dataclass
class RawAqua(RawMODIS):
    """
    Represents a class for processing raw Aqua MODIS data.

    Attributes:
        filename_regex (str): Regular expression pattern for matching Aqua MODIS filenames.
        satellite (str): Name of the satellite ("aqua").
        satellite_id (str): ID of the satellite ("MYD021KM").
    """

    filename_regex: str = (
        r"^MYD021KM\.A(?P<year>\d{4})(?P<day_of_year>\d{3})\.(?P<time>\d{4})\.(?P<version>\d+)\.(?P<timestamp>\d+)\.hdf"
    )
    satellite: str = "aqua"
    satellite_id: str = "MYD021KM"
    
    def load_raw_files(self, ifile):
        """
        Loads raw Aqua MODIS files.

        Args:
            ifile (str): Path to the input file.

        Returns:
            object: Raw MODIS bands data.
        """
        return load_modis_bands_raw(ifile, calibration=self.calibration)


@dataclass
class RawTerraCloud(RawMODIS):
    """
    Represents a class for loading raw Terra cloud data from MODIS dataset.
    """

    filename_regex: str = (
        r"^MYD35_L2\.A(?P<year>\d{4})(?P<day_of_year>\d{3})\.(?P<time>\d{4})\.(?P<version>\d+)\.(?P<timestamp>\d+)\.hdf"
    )
    satellite: str = "aqua_cloud"
    satellite_id: str = "MYD35_L2"
    
    def load_raw_files(self, ifile):
        """
        Loads raw files for Terra cloud data from the given input file.

        Args:
            ifile (str): The input file path.

        Returns:
            The loaded MODIS cloud mask raw data.
        """
        return load_modis_cloud_mask_raw(ifile)


@dataclass
class RawTerraCloud(RawMODIS):
    """
    A class representing raw Terra MODIS cloud data.

    Attributes:
        filename_regex (str): Regular expression pattern for matching the filename format.
        satellite (str): Name of the satellite (Terra).
        satellite_id (str): ID of the satellite (MOD35_L2).
    """

    filename_regex: str = (
        r"^MOD35_L2\.A(?P<year>\d{4})(?P<day_of_year>\d{3})\.(?P<time>\d{4})\.(?P<version>\d+)\.(?P<timestamp>\d+)\.hdf"
    )
    satellite: str = "terra_cloud"
    satellite_id: str = "MOD35_L2"
    
    def load_raw_files(self, ifile):
        """
        Load raw MODIS cloud mask data from a file.

        Args:
            ifile (str): Path to the input file.

        Returns:
            ndarray: The loaded MODIS cloud mask data.
        """
        return load_modis_cloud_mask_raw(ifile)
