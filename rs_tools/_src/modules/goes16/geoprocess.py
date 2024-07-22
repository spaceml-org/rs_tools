from typing import Optional, List, Iterable, Tuple, Callable, Any, cast
from rtree.index import Index, Property
from pathlib import Path
import pandas as pd
from rasterio.errors import RasterioIOError
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.polygon import Polygon
from rasterio.crs import CRS
import rioxarray
import xarray as xr
import numpy as np
import re
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from odc.geo.geom import BoundingBox, Geometry
# from torchgeo.datasets.utils import BoundingBox, disambiguate_timestamp
from rs_tools._src.modules.goes16.io import GOES16FileOrg, GOESFileName
from rs_tools._src.modules.goes16.ops import load_raw_single_band, load_raw_stacked_band
from rs_tools._src.modules.goes16.meta import GOES16_BANDS_TO_WAVELENGTHS
from rs_tools._src.preprocessing.nans import check_nan_count
from rs_tools._src.geoprocessing.reproject import calculate_latlon
from rs_tools._src.utils.io import get_list_filenames
from loguru import logger

GOES16_BANDS = list(map(lambda x: f"C{x:02}", range(1,17)))


@dataclass
class RawGoes16:
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
    filename_regex: str = r"^(?P<system>\D{2})_ABI-L1b-RadF-M6(?P<band>\w{3})_(?P<satellite>\w{3})_s(?P<date>\d{13})\d{1}_e\d{14}_c\d{14}"
    date_format: str = "%Y%j%H%M%S"
    bands: List[str] = field(default_factory=lambda: GOES16_BANDS)
    fill_value: float = 0.0
    transforms: Callable[[Any], Any] | None = None
    crs: CRS | None = None
    bbox: BoundingBox | None = None
    res: Tuple[float, float] | float | None = None
    
    def __post_init__(self):
        """
        Initializes the dataset by processing the files and creating a GeoDataFrame.
        """
        
        filename_regex = re.compile(self.filename_regex,re.VERBOSE)
        
        # get all files
        files = get_list_filenames(self.file_dir, self.file_ext)
        
        geo_dataframes = []
        
        for ifile in files:
            
            try:
                ifile_name = Path(ifile).name
                
                # match with regex
                match = re.match(filename_regex, str(ifile_name))
                
                if not match:
                    continue
                
                date = match.group("date")
                timestamp = pd.to_datetime(date, format=self.date_format)
                
                band = match.group("band")
                if band not in self.bands:
                    continue
                satellite = match.group("satellite")

                
                # open file
                xdset = rioxarray.open_rasterio(filename=ifile)
                
                
                if self.crs is None:
                    self.crs = xdset.rio.crs
                
                res = xdset.rio.resolution()
                if self.res is None:
                    self.res = res
                    
                if self.bbox is None:
                    self.bbox = BoundingBox(*xdset.rio.bounds(), crs=self.crs)
                
                # get polygon
                polygon = box(*self.bbox.bbox)
                
                # create pandas geodateframe
                df = pd.DataFrame(
                    {
                        "time": timestamp,
                        "satellite_id": satellite,
                        "band": band,
                        "full_path": str(ifile),
                    }, index=[0]
                )
                
                # convert the list of polygons into a GeoSeries
                geometry = gpd.GeoSeries(polygon, crs=self.crs)
                
                # create a GeoDataFrame
                geo_dataframes.append(
                    gpd.GeoDataFrame(df, geometry=geometry, crs=self.crs)
                )
            except RasterioIOError:
                continue
            
        # concatenate dateframes
        geo_dataframes = pd.concat(geo_dataframes, ignore_index=True)

        # clean and save the dataframe
        geo_dataframes = geo_dataframes.drop_duplicates().reset_index(drop=True)
        
        # drop columns that arent in bands
        geo_dataframes = geo_dataframes[geo_dataframes['band'].isin(self.bands)]
        
        # create an index
        time_index = pd.factorize(geo_dataframes["time"])[0]
        
        self.time_index = list(np.unique(time_index))
        
        geo_dataframes["time_index"] = time_index
        
        # save geodataframes
        self.geo_dataframes = geo_dataframes

    def __str__(self):
        """
        Returns a string representation of the dataset.
        """
        return (
            "GOES16 RAW Dataset" +
            f"\nCRS: {self.crs.to_proj4()}" +
            f"\nResolution: {self.res}" +
            f"\nBoundingBox: {self.bbox.bbox}" +
            f"\nNumber Files: {len(self)}" +
            f"\nNumber TimeStamps: {len(self.time_stamps)}" +
            f"\nBands: {self.bands}" +
            f"\nFill Value: {self.fill_value}" +
            f"\nTransforms: {self.transforms}" +
            f"\nFile Dir: {self.file_dir}" +
            f"\nFile Ext: {self.file_ext}"
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
        return load_raw_single_band(
            full_path, 
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


@dataclass
class RawGoes16Stack(RawGoes16):
    
    post_transforms: Callable[[Any], Any] | None = None
    
    def __str__(self):
        """
        Returns a string representation of the dataset.
        """
        return (
            "GOES16 Stacked Raw Dataset" +
            f"\ncrs: {self.crs}"+
            f"\nresolution: {self.res}"+
            f"\nNumber Files: {len(self)}"+
            f"\nBands: {self.bands}"
        )
    
    def __repr_(self):
        """
        Returns a string representation of the dataset.
        """
        return (
             "GOES16 Stacked Raw Dataset" +
            f"\ncrs: {self.crs}" +
            f"\nresolution: {self.res}" +
            f"\nNumber Files: {len(self)}" +
            f"\nBands: {self.bands}"
        )
        
    def __len__(self):
        return len(self.time_index)
    
    def __getitem__(self, idx):
        
        # filter dataframe for files
        filtered_df = self.geo_dataframes[self.geo_dataframes["time_index"] == idx]
        
        full_paths = list(filtered_df["full_path"])
        
        stacked_ds = load_raw_stacked_band(
            full_paths=full_paths,
            bbox=self.bbox,
            crs=self.crs,
            res=self.res,
            fill_value=self.fill_value,
            transforms=self.transforms,
            post_transforms=self.post_transforms,
        )
        
        return stacked_ds


def create_metafile_from_regex(
    file_dir: str,
    file_est: str,
    filename_regex: str,
    date_format: str,
    crs: CRS | None = None,
    bbox: BoundingBox | None = None,
    res: Tuple[float, float] | None = None,
    ) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame containing metadata from files that match a given filename regex pattern.

    Args:
        files (List[str]): List of file paths.
        filename_regex (str): Regular expression pattern to match the file names.
        date_format (str): Format of the date in the file names.
        crs (CRS | None, optional): Coordinate reference system. Defaults to None.
        bbox (BoundingBox | None, optional): Bounding box. Defaults to None.
        res (Tuple[float, float] | None, optional): Resolution. Defaults to None.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the metadata.

    Raises:
        RasterioIOError: If there is an error reading the raster file.

    """
    filename_regex = re.compile(filename_regex,re.VERBOSE)
    
    # get all files
    files = get_list_filenames(file_dir, file_ext)
    
    geo_dataframes = []
    
    for ifile in files:
        
        try:
            ifile_name = Path(ifile).name
            
            # match with regex
            match = re.match(filename_regex, str(ifile_name))
            
            date = match.group("date")
            timestamp = pd.to_datetime(date, format=date_format)
            
            band = match.group("band")
            satellite = match.group("satellite")

            
            # open file
            xdset = rioxarray.open_rasterio(filename=ifile)
            
            crs = xdset.rio.crs
            if crs is None:
                crs = crs
            
            res = xdset.rio.resolution()
            if res is None:
                res = res
                
            if bbox is None:
                bbox = BoundingBox(*xdset.rio.bounds(), crs=crs)
            
            # get polygon
            polygon = box(*xdset.rio.bounds())
            
            # create pandas geodateframe
            df = pd.DataFrame(
                {
                    "time": timestamp,
                    "satellite_id": satellite,
                    "band": band,
                    "full_path": str(ifile),
                }, index=[0]
            )
            
            # convert the list of polygons into a GeoSeries
            geometry = gpd.GeoSeries(polygon)
            
            # create a GeoDataFrame
            geo_dataframes.append(
                gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
            )
        except RasterioIOError:
            continue
        
    # concatenate dateframes
    geo_dataframes = pd.concat(geo_dataframes, ignore_index=True)

    # clean and save the dataframe
    geo_dataframes = geo_dataframes.drop_duplicates()
    
    return geo_dataframes

