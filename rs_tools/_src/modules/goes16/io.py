from typing import List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import rioxarray
import xarray as xr
import pandas as pd
import geopandas as gpd
from pathlib import PosixPath
from pandas._libs.tslibs.timestamps import Timestamp
from rs_tools._src.utils.io import get_list_filenames
from odc.geo.geom import BoundingBox
from odc.geo.crs import CRS


from rasterio.errors import RasterioIOError


@dataclass
class GOES16FileOrg:
    """
    A class representing the organization of GOES-16 files.
    
    Attributes:
        read_dir (PosixPath): The directory path where the files are located.
    """
    read_dir: PosixPath
    
    def __str__(self):
        return (
            "GOES16 Filer:"+
            f"\nNum Files: {self.num_files}"+
            f"\nNum File Sets: {self.num_filesets}" +
            f"\nDirectory: {self.read_dir}"
            )
        
    def __repr__(self):
        return self.__str__()
    
    @classmethod
    def init_from_str(cls, read_dir: str):
        """
        Initializes a GOES16FileOrg object from a directory path.
        
        Args:
            read_dir (str): The directory path where the files are located.
        
        Returns:
            GOES16FileOrg: The initialized GOES16FileOrg object.
        
        Raises:
            AssertionError: If the specified directory does not exist.
        """
        # check if directory exists
        read_dir = Path(read_dir)
        assert read_dir.is_dir()
        return cls(read_dir=read_dir)
    
    @property
    def all_files(self):
        """
        Returns a list of all files in the specified directory with the extension ".nc".
        
        Returns:
            List[str]: A list of file names.
        """
        return get_list_filenames(self.read_dir, ".nc")
    
    def __len__(self):
        """
        Returns the number of files in the specified directory.
        
        Returns:
            int: The number of files.
        """
        return len(self.all_files)
    

    
    @property
    def unique_times(self) -> List[str]:
        """
        Returns a list of unique times extracted from the file names.
        
        Returns:
            List[str]: A list of unique time strings.
            
        """
        fn = lambda x: GOESFileName.init_from_filename(x).time_start
        return list(set(map(fn, self.all_files)))
    
    def iter_files(self):
        for ifile in self.all_files:
            yield GOESFileName.init_from_filename(ifile)
    
    @property
    def files(self):
        return list(self.iter_files())
    
    @property
    def num_files(self):
        """
        Returns the number of files in the specified directory.
        
        Returns:
            int: The number of files.
        """
        return len(self.files)
    
    def iter_fileset(self):
        """
        Iterates over the unique times and yields a list of files for each time.
        
        Yields:
            List[str]: A list of file names for each unique time.
        """
        for itime in self.unique_times:
            # search in files for time stamp
            itime = goes_timestamp_to_datestr(itime)
            files = list(filter(lambda x: itime in x, self.all_files))
            fn = lambda x: GOESFileName.init_from_filename(x)
            yield {str(itime):list(set(map(fn, files)))}
            
    @property
    def fileset(self):
        return list(self.iter_fileset())

    @property
    def num_filesets(self):
        """
        Returns the number of files in the specified directory.
        
        Returns:
            int: The number of files.
        """
        return len(self.fileset)


@dataclass(order=True, frozen=True)
class GOESFileName:
    """
    Represents a GOES file name and provides methods to parse the file name components.
    """

    full_path: Path
    operation_system: str
    satellite_id: str
    imager: str
    level: str
    product_id: str
    abi_mode: str
    channel: str
    satellite: str
    time_start: Timestamp
    time_end: Timestamp
    time_creation: Timestamp
    
    @property
    def channel_id(self):
        return int(self.channel[-1:])
    
    @classmethod
    def init_from_filename(cls, file_path: str):
        """
        Initializes a GOESFileName object from a file path.

        Args:
            file_path (str): The path of the GOES file.

        Returns:
            GOESFileName: The initialized GOESFileName object.
        """
        
        full_path = Path(file_path)
        # parse filename
        file_name = full_path.stem
        # replace the hyphens with under-braces
        file_name = file_name.replace("-", "_")
        # split by under-braces
        file_name_parts = file_name.split("_")
        # assign bits n pieces
        operation_system = file_name_parts[0]
        imager = file_name_parts[1]
        level = file_name_parts[2]
        product_id = file_name_parts[3]
        abi_mode = file_name_parts[4][:2]
        channel = file_name_parts[4][2:]
        satellite_id = file_name_parts[5]
        time_start = goes_datestr_to_timestamp(file_name_parts[6][1:-1])
        time_end = goes_datestr_to_timestamp(file_name_parts[7][1:-1])
        time_creation = goes_datestr_to_timestamp(file_name_parts[8][1:-1])
        return cls(
            full_path=full_path,
            operation_system=operation_system,
            imager=imager,
            satellite_id=satellite_id,
            level=level,
            product_id=product_id,
            abi_mode=abi_mode,
            channel=channel,
            satellite=satellite_id,
            time_start=time_start,
            time_end=time_end,
            time_creation=time_creation,
        )

    def __str__(self):
        return (
            "GOES File:" +
            f"\nSatellite: {self.satellite_id}" +
            f"\nLevel: {self.level}" +
            f"\nProduct: {self.product_id}" +
            f"\nChannel: {self.channel}" +
            f"\nTime: {self.time_start}" +
            f"\nFile: {Path(self.full_path).name}"
            f"\nParent Dir: {Path(self.full_path).parent}"
        )
        
    def __repr__(self):
        return self.__str__()


def goes_datestr_to_timestamp(datestr: str) -> Timestamp:
    """
    Converts a GOES date string to a pandas Timestamp object.

    Parameters:
        datestr (str): The GOES date string in the format "%Y%j%H%M%S%s".

    Returns:
        Timestamp: A pandas Timestamp object representing the converted date and time.
    """
    return pd.to_datetime(datestr, format="%Y%j%H%M%S")


def goes_timestamp_to_datestr(timestamp: Timestamp) -> str:
    """
    Converts a pandas Timestamp object to a string representation of the date in the format '%Y%j%H%M%S'.

    Parameters:
        timestamp (pandas.Timestamp): The timestamp to convert.

    Returns:
        str: The string representation of the date.

    Example:
        >>> timestamp = pd.Timestamp('2022-01-01 12:34:56')
        >>> goes_timestamp_to_datestr(timestamp)
        '2022001123456'
    """
    return timestamp.strftime('%Y%j%H%M%S')


def parse_goes16_dates_from_file(file: str):
    """
    Parses the date and time information from a GOES-16 file name.

    Args:
        file (str): The file name to parse.

    Returns:
        str: The parsed date and time in the format 'YYYYJJJHHMM'.
    """
    timestamp = Path(file).name.replace("-", "_").split("_")
    
    return datetime.strptime(timestamp[-3][1:], "%Y%j%H%M%S%f").strftime("%Y%j%H%M%S")


def format_goes_dates(time: str) -> str:
    """
    Function to format the date/time string.
    
    Args:
        time (str): The time string to be formatted.
        
    Returns:
        str: The formatted time string.
    """

    # Split the string into date and time parts
    date_str, time_str = time[:7], time[7:]
    # Convert the date part to a datetime object
    date = datetime.strptime(date_str, "%Y%j")

    # Convert the time part to a timedelta object
    time = timedelta(hours=int(time_str[:2]), minutes=int(time_str[2:4]), seconds=int(time_str[4:6]))
    # Add the date and time parts to get a datetime object
    dt = date + time
    # Convert the datetime object to a string in the format "YYYYMMDDHHMMSS"
    str_time = dt.strftime("%Y%m%d%H%M%S")
    
    return str_time


def create_metafile_from_directory(
    file_dir: str,
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
    
    files = get_list_filenames(file_dir, ".nc")
    

    
    geo_dataframes = []
    
    
    for ifile in files:
        
        try:
            
            goes_file_obj = GOESFileName.init_from_filename(str(ifile))
            
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
            polygon = BoundingBox(*xdset.rio.bounds(), crs=crs).polygon
            
            # create pandas geodateframe
            df = pd.DataFrame(
                {
                    "time": goes_file_obj.time_start,
                    "satellite_id": goes_file_obj.satellite_id,
                    "band": goes_file_obj.channel_id,
                    "full_path": str(ifile),
                }, index=[0]
            )
            
            # convert the list of polygons into a GeoSeries
            
            geometry = gpd.GeoSeries([polygon])
            
            # create a GeoDataFrame
            geo_dataframes.append(
                gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
            )
        except RasterioIOError:
            continue
            
        
    # concatenate dateframes
    geo_dataframes = pd.concat(geo_dataframes, ignore_index=True)

    # clean and save the dataframe
    geo_dataframes = geo_dataframes.drop_duplicates().reset_index(drop=True)
    
    return geo_dataframes
