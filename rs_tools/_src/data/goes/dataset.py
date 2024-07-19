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
from rs_tools._src.data.goes.io import GOES16FileOrg, GOESFileName
from rs_tools._src.data.goes.bands import GOES16_BANDS_TO_WAVELENGTHS
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

    Methods:
        __post_init__(): Initializes the dataset by processing the files and creating a GeoDataFrame.
        __str__(): Returns a string representation of the dataset.
        __repr__(): Returns a string representation of the dataset.
        __len__(): Returns the number of images in the dataset.
        period(): Returns the bounding box of the dataset.
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
    
    def __post_init__(self):
        """
        Initializes the dataset by processing the files and creating a GeoDataFrame.
        """
        self.src_crs = None
        self.src_bbox = None
        self.src_res = None
        
        filename_regex = re.compile(self.filename_regex,re.VERBOSE)
        
        # get all files
        files = get_list_filenames(self.file_dir, self.file_ext)
        
        geo_dataframes = []
        
        for ifile in files:
            
            try:
                ifile_name = Path(ifile).name
                
                # match with regex
                match = re.match(filename_regex, str(ifile_name))
                
                date = match.group("date")
                timestamp = pd.to_datetime(date, format=self.date_format)
                
                band = match.group("band")
                if band not in self.bands:
                    continue
                satellite = match.group("satellite")

                
                # open file
                xdset = rioxarray.open_rasterio(filename=ifile)
                
                crs = xdset.rio.crs
                if self.src_crs is None:
                    self.src_crs = crs
                
                res = xdset.rio.resolution()
                if self.src_res is None:
                    self.src_res = res
                    
                if self.src_bbox is None:
                    self.src_bbox = BoundingBox(*xdset.rio.bounds(), crs=crs)
                
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
        
        # drop columns that arent in bands
        # geo_dataframes = geo_dataframes[geo_dataframes['band'].isin(self.bands)]
        
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
            f"\nCRS: {self.src_crs.to_proj4()}" +
            f"\nResolution: {self.src_res}" +
            f"\nBoundingBox: {self.src_bbox.bbox}" +
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
            full_path, bbox=self.src_bbox,
            crs=self.src_crs,
            res=self.src_res,
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
            # filter dataframe for files
            xdset = self.iloc(idx)
            
            if self.transforms:
                xdset = self.transforms(xdset)
            
            yield xdset
        
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
        
        bands = []
        band_wavelengths = []
        for i, (irow, iframe) in enumerate(filtered_df.iterrows()):
            # open dataset
            xdset = load_raw_single_band(
                iframe["full_path"],
                bbox=self.src_bbox,
                crs=self.src_crs,
                res=self.src_res,
                fill_value=self.fill_value,
                )
            bands.append(xdset.band.values.squeeze())
            band_wavelengths.append(xdset.band_wavelength.values.squeeze())
            band_wl_attrs = xdset.band_wavelength.attrs
            band_attrs = xdset.band.attrs
            # apply transforms
            if self.transforms:
                xdset = self.transforms(xdset)
            
            # concatenation
            if i == 0:
                stacked_ds = xdset
                
            else:
                xdset = xdset.interp(x=stacked_ds.x, y=stacked_ds.y)
                stacked_ds = xr.concat([stacked_ds, xdset], dim="band")
            
            # del xdset
            
        # reassign bands
        dims = stacked_ds.dims
        stacked_ds = stacked_ds.assign_coords({"band": bands})
        stacked_ds["band"].attrs = band_attrs
        stacked_ds = stacked_ds.assign_coords({"band_wavelength": (("band"), band_wavelengths)})
        stacked_ds["band_wavelength"].attrs = band_wl_attrs
        if "time" not in dims:
            stacked_ds = stacked_ds.expand_dims({"time": [iframe["time"]]})
            
        # apply transforms
        if self.post_transforms:
            stacked_ds = self.post_transforms(stacked_ds)
        
        return stacked_ds


def create_metafile_from_regex(
    files: List[str],
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
    filename_regex = re.compile(self.filename_regex,re.VERBOSE)
    
    crs = None
    bbox = None
    res = None
    
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




def load_raw_single_band(
    full_path,
    bbox: BoundingBox | None = None,
    crs: CRS | None = None,
    res: float | None = None,
    resampling: Resampling = Resampling.bilinear,
    fill_value: float = 0.0,
    ) -> xr.Dataset:
    """
    Load a single band from a dataset.

    Args:
        full_path (str): The full path to the dataset.
        bbox (BoundingBox | None, optional): The bounding box coordinates (left, bottom, right, top) to clip the data. Defaults to None.
        crs (CRS | None, optional): The coordinate reference system to use for clipping and reprojection. Defaults to None.
        res (float | None, optional): The desired resolution for reprojection. Defaults to None.
        resampling (Resampling, optional): The resampling method to use for reprojection. Defaults to Resampling.bilinear.
        fill_value (float, optional): The value to use for filling missing or masked data. Defaults to 0.0.

    Returns:
        xr.Dataset: The loaded band as a Dataset object.

    Raises:
        ValueError: If the filename does not conform to the GOES16 Standard.

    """
    
    try:
        goes_filename = GOESFileName.init_from_filename(full_path)
        band = goes_filename.channel
        time = goes_filename.time_start
    except ValueError:
        msg = f"Unrecognized filename: \n{Path(full_path).name}"
        msg += f"\nDoes not conform to GOES16 Standard..."
        raise ValueError(msg)
    # open the dataset
    xdset = rioxarray.open_rasterio(full_path)
    
    if crs is None:
        crs = xdset.rio.crs
    
    # clip data
    if bbox is not None:
        xdset = xdset.rio.clip_box(*bbox.bbox, crs=bbox.crs)

    # reproject
    if res is not None: 
        xdset = xdset.rio.reproject(dst_crs=crs, resolution=res, resampling=resampling)
        
    # extract meta-data from filename
    
    
    # add the band dimensions
    xdset = xdset.assign_coords({"band": [band]})
    xdset["band"].attrs = {
        "long_name": "ABI band number",
        "standard_name": "sensor_band_identifier"
    }
    
    # add the time dimension
    xdset = xdset.expand_dims({"time": [time]})
    
    band_str = str(int(band[1:]))
    xdset = xdset.assign_coords({"band_wavelength": (("band"), [GOES16_BANDS_TO_WAVELENGTHS[band_str]])})
    xdset["band_wavelength"].attrs = {
        "long_name": "ABI band central wavelength",
        "standard_name": "sensor_band_central_radiation_wavelength",
        "units": "um"
    }
    
    # convert fill value
    xdset["Rad"] = xdset["Rad"].where(xdset["Rad"] != xdset["Rad"].attrs["_FillValue"], fill_value)
    
    # fix attributes
    keys = ["long_name", "standard_name", "valid_range", "units", "scale_factor", "add_offset", "fill_value"]
    keep_attrs = {key: xdset.Rad.attrs[key] for key in keys if key in xdset["Rad"].attrs.keys()}
    xdset["Rad"].attrs = {}
    xdset["Rad"].attrs = keep_attrs
    
    # move the data-quality flags to coordinates
    xdset = xdset.drop_vars("DQF")
    
    # just in case...
    xdset.rio.write_crs(crs, inplace=True)

    return xdset.Rad


def save_g16_band_to_tiff(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool=False,
        save_latlon: bool=True,
        fill_value: float=0.0
):  
    """
    Save a GOES-16 band to a TIFF file.

    Parameters:
        ds (xr.DataArray): The input data array containing the band data.
        save_dir (str): The directory where the TIFF file will be saved.
        overwrite (bool, optional): Whether to overwrite the file if it already exists. Defaults to False.
        save_latlon (bool, optional): Whether to save latitude and longitude coordinates. Defaults to True.
        fill_value (float, optional): The fill value to use for NaN values. Defaults to 0.0.

    Returns:
        None
    """
    ifile_path = Path(save_dir)
    
    # assert directory exists
    ifile_path.mkdir(parents=True, exist_ok=True)
    
    # grab time components
    time_stamp = pd.to_datetime(ds.time.values.squeeze())
    save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
    band = ds.band.values
    if len(band) > 1:
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_rads.tiff")
    else:
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_{band.squeeze()}_rads.tiff")

    if overwrite and ifile_path.is_file():
        # remove file if it already exists
        ifile_path.unlink()
    # select variables
    
    crs = ds.rio.crs
    ds = ds.where(ds != np.nan, fill_value)
    # remove all extra dimensions
    ds = ds.squeeze()
    
    logger.info(f"Dims: {ds.dims}")
    # assert len(ds.dims) == 3
    
    
    
    # add coordinates
    if save_latlon:
        ds = ds.to_dataset(dim="band")
        LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
        ds["lat"] = (("x", "y", "band"), LATS[..., None])
        ds["lon"] = (("x", "y", "band"), LONS[..., None])
        ds = ds.to_array(dim="band")
        
        
    # change long name for rasterio to understand the band descriptions...
    try:
        ds.attrs["long_name"] = list(map(lambda x: str(x), ds.band.values))
    except TypeError:
        ds.attrs["long_name"] = str(ds.band.values)
    logger.info(f"long_name: {ds.attrs['long_name']}")

    # save with rasterio
    ds.rio.to_raster(ifile_path)

    return None
    

def save_g16_to_tiff(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool=False,
        save_latlon: bool=True,
        fill_value: float=0.0
):  
        ifile_path = Path(save_dir)
        
        # assert directory exists
        ifile_path.mkdir(parents=True, exist_ok=True)
        
        # grab time components
        time_stamp = pd.to_datetime(ds.time.values.squeeze())
        save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
        
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_rads.tiff")

        if overwrite and ifile_path.is_file():
            # remove file if it already exists
            ifile_path.unlink()
        # select variables
        
        crs = ds.rio.crs
        ds = ds
        ds = ds.where(ds != np.nan, fill_value)
        # create a dataset
        ds = ds.to_dataset(dim="band")
        
        ds.rio.write_crs(crs, inplace=True)

        # remove attributes (otherwise it doesnt save properly...)
        ds.attrs = {}

        # add coordinates
        if save_latlon:
            LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
            ds["lat"] = (("x", "y"), LATS)
            ds["lon"] = (("x", "y"), LONS)
        
        # save with rasterio
        ds.squeeze().rio.to_raster(ifile_path)

        return None
    

    
def save_g16_band_to_npy(
        ds: xr.DataArray,
        save_dir: str,
        overwrite: bool=False,
        save_latlon: bool=True,
        fill_value: float=0.0
):  
        ifile_path = Path(save_dir)
        
        # assert directory exists
        ifile_path.mkdir(parents=True, exist_ok=True)
        
        # grab time components
        time_stamp = pd.to_datetime(ds.time.values.squeeze())
        save_file_name = datetime.strftime(time_stamp, "%Y%m%d%H%M%S")
        band = ds.band.values.squeeze()
        
        ifile_path = ifile_path.joinpath(f"{save_file_name}_g16_{band}_rads.tiff")

        if overwrite and ifile_path.is_file():
            # remove file if it already exists
            ifile_path.unlink()
        # select variables
        
        crs = ds.rio.crs
        ds = ds.Rad
        ds = ds.where(ds != np.nan, fill_value)
        # create a dataset
        ds = ds.to_dataset(dim="band")
        
        ds.rio.write_crs(crs, inplace=True)

        # remove attributes (otherwise it doesnt save properly...)
        ds.attrs = {}

        # add coordinates
        if save_latlon:
            LONS, LATS = calculate_latlon(ds.x.values, ds.y.values, crs=ds.rio.crs)
            ds["lat"] = (("x", "y"), LATS)
            ds["lon"] = (("x", "y"), LONS)
        
        # save with rasterio
        ds.squeeze().rio.to_raster(ifile_path)

        return None