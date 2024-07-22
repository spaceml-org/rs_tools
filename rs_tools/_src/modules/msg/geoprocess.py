from dataclasses import (
    dataclass,
    field,
)
from datetime import (
    datetime,
    timedelta,
)
from pathlib import Path
import re
import sys
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    cast,
)

import geopandas as gpd
from loguru import logger
import numpy as np
from odc.geo.geom import (
    BoundingBox,
    Geometry,
)
import pandas as pd
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
import rioxarray
from rtree.index import (
    Index,
    Property,
)
from shapely.geometry import box
from shapely.geometry.polygon import Polygon
import xarray as xr

from rs_tools._src.modules.msg.meta import MSG_BANDS
from rs_tools._src.modules.msg.ops import (
    add_msg_crs,
    clean_msg_rad_raw_satpy,
    load_msg_rads_raw_satpy,
)
from rs_tools._src.utils.io import get_list_filenames


@dataclass
class RawMSGL1Radiance:
    """
    Represents a dataset of raw GOES-16 satellite images.

    Attributes:
        files (List[str]): List of file paths for the dataset.
        filename_regex (str): Regular expression pattern for matching file names.
        date_format (str): Format string for parsing the date from file names.
        bands (List[str]): List of bands to include in the dataset.
        fill_value (float): Fill value for missing data.
        transforms (Callable[[Any], Any] | None): Optional data transformation function.
        crs (CRS | None): Coordinate reference system of the dataset.
        bbox (BoundingBox | None): Bounding box of the dataset.
        res  (Tuple[float, float] | float | None): Resolution of the dataset.

    Methods:
        __post_init__(): Initializes the dataset by processing the files and creating a GeoDataFrame.
        __str__(): Returns a string representation of the dataset.
        __repr__(): Returns a string representation of the dataset.
        __len__(): Returns the number of images in the dataset.
        time_stamps(): Returns a list of unique timestamps in the dataset.
        total_indices(): Returns the total number of indices in the dataset.
        iloc(idx): Retrieves a single band from the dataset based on the given index.
        generator(): Generates the dataset by iterating over the bands.
        __getitem__(idx): Retrieves a single band from the dataset based on the given index.
    """

    file_dir: str
    file_ext: str = ".nc"
    filename_regex: str = r"^MSG4-SEVI-MSG15-\d{4}-NA-(?P<date>\d{14})\.\d{9}"
    date_format: str = "%Y%m%d%H%M%S"
    bands: List[str] = field(default_factory=lambda: MSG_BANDS)
    fill_value: float = 0.0
    transforms: Callable[[Any], Any] | None = None
    crs: CRS | None = None
    bbox: BoundingBox | None = None
    res: Tuple[float, float] | float | None = None

    def __post_init__(self):
        """
        Initializes the dataset by processing the files and creating a GeoDataFrame.
        """

        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        # get all files
        files = get_list_filenames(self.file_dir, self.file_ext)

        geo_dataframes = []

        res = self.res
        crs = self.crs
        bbox = self.bbox

        for ifile in files:

            try:
                ifile_name = Path(ifile).name

                # match with regex
                match = re.match(filename_regex, str(ifile_name))

                if not match:
                    continue

                date = match.group("date")
                timestamp = pd.to_datetime(date, format=self.date_format)

                # open file
                xdset = load_msg_rads_raw_satpy(file=ifile)
                # add crs
                xdset = add_msg_crs(xdset)

                if crs is None:
                    crs = xdset.rio.crs

                if res is None:
                    res = xdset.rio.resolution()

                if bbox is None:
                    bbox = BoundingBox(*xdset.rio.bounds(), crs=crs)

                # get polygon
                polygon = box(*bbox.bbox)

                # create pandas geodateframe
                df = pd.DataFrame(
                    {
                        "time": timestamp,
                        "satellite_id": "msg",
                        "full_path": str(ifile),
                    },
                    index=[0],
                )

                # convert the list of polygons into a GeoSeries
                geometry = gpd.GeoSeries(polygon, crs=crs)

                # create a GeoDataFrame
                geo_dataframes.append(gpd.GeoDataFrame(df, geometry=geometry, crs=crs))
            except RasterioIOError:
                continue

        # concatenate dateframes
        geo_dataframes = pd.concat(geo_dataframes, ignore_index=True)

        # clean and save the dataframe
        geo_dataframes = geo_dataframes.drop_duplicates().reset_index(drop=True)

        # create an index
        time_index = pd.factorize(geo_dataframes["time"])[0]

        self.time_index = list(np.unique(time_index))

        geo_dataframes["time_index"] = time_index

        # save geodataframes
        self.geo_dataframes = geo_dataframes
        self.bbox = bbox
        self.crs = crs

    def __str__(self):
        """
        Returns a string representation of the dataset.
        """
        return (
            "GOES16 RAW Dataset"
            + f"\nCRS: {self.crs.to_proj4()}"
            + f"\nResolution: {self.res}"
            + f"\nBoundingBox: {self.bbox.bbox}"
            + f"\nNumber Files: {len(self)}"
            + f"\nNumber TimeStamps: {len(self.time_stamps)}"
            + f"\nBands: {self.bands}"
            + f"\nFill Value: {self.fill_value}"
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

    def iloc(self, idx) -> xr.Dataset:
        """
        Retrieves a single band from the dataset based on the given index.

        Parameters:
            idx (int): The index of the band to retrieve.

        Returns:
            ndarray: The single band data.
        """
        full_path = self.geo_dataframes.iloc[idx]["full_path"]
        # load raw file
        ds = load_msg_rads_raw_satpy(full_path)
        return clean_msg_rad_raw_satpy(
            ds,
            bbox=self.bbox,
            crs=self.crs,
            res=self.res,
            fill_value=self.fill_value,
        )

    def generator(self):
        """
        Generates the dataset by iterating over the bands.

        Yields:
            xr.Dataset: The band data.
        """
        num_iterations = len(self)

        for idx in range(num_iterations):

            yield self.__getitem__(idx)

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
