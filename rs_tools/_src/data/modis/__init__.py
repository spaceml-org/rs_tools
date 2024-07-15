import os
from typing import List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
import earthaccess
import earthaccess.results
from rs_tools._src.utils.io import get_list_filenames
from odc.geo.geom import Geometry, polygon


# TODO: Expand mapping to other resolutions (250m, 500m)
MODIS_NAME_TO_ID= dict(
    terra="MOD021KM",
    terra_geo="MOD03",
    terra_cloud="MOD35_L2",
    aqua="MYD021KM",
    aqua_geo="MYDO3",
    aqua_cloud="MYD35_L2",
)

# TODO: Expand mapping to other resolutions (250m, 500m)
MODIS_ID_TO_NAME = dict(
    MYD021KM="aqua",
    MYDO3="aqua_geo",
    MYD35_L2="aqua_cloud",
    MOD021KM="terra",
    MOD03="terra_geo",
    MOD35_L2="terra_cloud",
)

@dataclass
class MODISRawFiles:
    read_dir: str

    @property
    def modis_files(self):
        return get_list_filenames(self.read_dir, ".hdf")

    def get_modis_files_from_satellite(self, satellite_name: str="aqua"):
        # get satellite ID
        satellite_id = MODIS_NAME_TO_ID[satellite_name]
        # filter files for aqua only
        return list(filter(lambda x: satellite_id in x, self.modis_files))

    @property
    def modis_file_obj(self):
        return list(map(lambda x: MODISFileName.from_filename(x), self.modis_files))

    @property
    def modis_file_obj_pairs(self):
        return get_modis_paired_files(self.modis_file_obj)


@dataclass(order=True, frozen=True)
class MODISFileName:
    """
    M[O/Y]D02[1/H/Q]KM.A[date].[time].[collection].[processing_time].hdf
    """
    save_path: str 
    satellite_id: str 
    collection: str
    ext: str
    datetime_acquisition: datetime
    datetime_processing: datetime

    @classmethod
    def from_filename(cls, file_name: str):
        """
        Creates a MODISFileName object from a given file name.

        Args:
            cls (type): The class object that the method is bound to.
            file_name (str): The file name to parse.

        Returns:
            MODISFileName: The parsed MODISFileName object.
        """

        file_name = Path(file_name)
        components = file_name.name.split('.')
        save_path = str(file_name.parents[0])
        satellite_id = components[0]
        collection = components[3]
        
        # time and date
        time = components[2]
        year = components[1][1:5]
        day = components[1][-3:]
        datetime_acquisition = datetime.strptime(f"{year}{day}{time}", "%Y%j%H%M")

        # processing time
        year = components[-2][:4]
        day = components[-2][4:7]
        hour = components[-2][-6:-4]
        minute = components[-2][-4:-2]
        second = components[-2][-2:]
        datetime_processing = datetime.strptime(f"{year}{day}{hour}{minute}{second}", "%Y%j%H%M%S")
       
        # extension
        ext = components[-1]
        return cls(
            save_path=save_path,
            satellite_id=satellite_id,
            collection=collection,
            datetime_acquisition=datetime_acquisition,
            datetime_processing=datetime_processing,
            ext=ext)

    @property
    def satellite_name(self):
        """
        Gets the name of the satellite based on the satellite ID.

        Returns:
            str: The name of the satellite.
        """
        return MODIS_ID_TO_NAME[self.satellite_id]
    
    @property
    def modis_filename(self):
        """
        Generates the MODIS file name based on the object's properties.

        Returns:
            str: The generated MODIS file name.
        """
        # satellite ID
        filename = f"{self.satellite_id}"
        # Date
        date = self.datetime_acquisition.strftime("%Y%j")
        filename += f".A{date}"
        # Time
        time = self.datetime_acquisition.strftime("%H%M")
        filename += f".{time}"
        # Collection
        filename += f".{self.collection}"
        # Processing Time
        time = self.datetime_processing.strftime("%Y%j%H%M%S")
        filename += f".{time}"
        filename += f".{self.ext}"
        return filename
    
    @property
    def full_path(self):
        """
        Gets the full path of the MODIS file.

        Returns:
            Path: The full path of the MODIS file.
        """
        return Path(self.save_path).joinpath(self.modis_filename)
    

# NOTE: we no longer download geo data
def get_modis_paired_files(files: List[MODISFileName], satellite="aqua"):
    # get satellite filenames
    modis_satellite = list(filter(lambda x: x.satellite_name == satellite, files))

    # get corresponding paired files
    paired = dict()
    for isatellite in modis_satellite:
        # get the time for the acquisition
        itime = isatellite.datetime_acquisition.strftime("%Y%m%d%H%M")
    
        ifile = dict(data=isatellite)
        # filter to be the same time
        criteria = lambda x: x.datetime_acquisition == isatellite.datetime_acquisition
        subset_files = list(filter(criteria, files))
        
        # find geolocation file
        try:
            criteria = lambda x: x.satellite_name == f"{satellite}_geo"
            ifile["geo"] = list(filter(criteria, subset_files))[0]
        except IndexError:
            pass
    
        # find cloudmask file
        try:
            criteria = lambda x: x.satellite_name == f"{satellite}_cloud"
            ifile["cloud"] = list(filter(criteria, subset_files))[0]
        except IndexError:
            pass
    
        paired[itime] = ifile

    return paired


def query_modis_timestamps(
        short_name: str,
        bounding_box: tuple,
        temporal: tuple,
        ):
    """
    Function to query the Earthdata API for MODIS data timestamps.
    :param short_name: MODIS short name (e.g. 'MOD021KM' for Terra MODIS Level 1B data at 1km resolution)
    :param bounding_box (tuple, optional): The region to be queried. Follows format (min_lon, min_lat, max_lon, max_lat).
    :param temporal: Min and max date/time to be queried. Follows format (start_datetime: YYYY-MM-DD HH:MM:SS, end_datetime: YYYY-MM-DD HH:MM:SS).
    :return: result object
    """
    results = earthaccess.search_data(
        short_name=short_name,
        cloud_hosted=True,
        bounding_box=bounding_box,
        temporal=temporal,
        count=-1
    )
    return results


def modis_granule_to_datetime(granule: earthaccess.results.DataGranule) -> datetime:
    """
    Function to convert a MODIS granule to a datetime object.
    :param granule: MODIS granule
    :return: datetime object
    """
    return datetime.strptime(granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")


def modis_granule_to_polygon(granule: earthaccess.results.DataGranule) -> Geometry:
    """
    Converts a MODIS granule to a polygon geometry.

    Args:
        granule (earthaccess.results.DataGranule): The MODIS granule to convert.

    Returns:
        Geometry: The polygon geometry representing the MODIS granule.
    """

    points_list = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
    coordinates = [(point['Longitude'], point['Latitude']) for point in points_list]
    return polygon(coordinates, crs="4326")


def modis_granule_to_satellite_id(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["CollectionReference"]["ShortName"]

def modis_granule_to_filename(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]


def modis_granule_to_satellite_name(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["Platforms"][0]["ShortName"].lower()


def modis_granule_to_instrument(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["Platforms"][0]["Instruments"][0]["ShortName"]

def modis_granule_to_daynightflay(granule: earthaccess.results.DataGranule) -> str:
    return granule["umm"]["DataGranule"]["DayNightFlag"]

def modis_granule_to_gdf(ea_granules):

    # gather all modis timestamps
    modis_timestamps = list(map(modis_granule_to_datetime, ea_granules))

    # gather all polygons
    modis_polygons = list(map(modis_granule_to_polygon, ea_granules))

    # gather satellite ids
    modis_satellite_ids = list(map(modis_granule_to_satellite_id, ea_granules))

    # gather satellite names
    modis_satellite_names = list(map(modis_granule_to_satellite_name, ea_granules))

    # gather satellite filenames
    modis_satellite_filenames = list(map(modis_granule_to_filename, ea_granules))

    # gather satellite filenames
    modis_satellite_instruments = list(map(modis_granule_to_instrument, ea_granules))

    # day
    modis_satellite_daynightflag = list(map(modis_granule_to_instrument, ea_granules))

    # create pandas geodataframe
    df = pd.DataFrame({
        "time": modis_timestamps,
        "satellite_id": modis_satellite_ids,
        "satellite_name": modis_satellite_names,
        "filename": modis_satellite_filenames,
        "daynight": modis_satellite_daynightflag,
        "satellite_instrument": modis_satellite_instruments,
        })

    # convert the list of polygons into a GeoSeries
    geometry = gpd.GeoSeries(modis_polygons)

    # create a GeoDataFrame 
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    return gdf


def parse_modis_dates_from_file(file: str):
    """
    Parses the date and time information from a MODIS file name.

    Args:
        file (str): The file name to parse.

    Returns:
        str: The parsed date and time in the format 'YYYYJJJHHMM'.
    """
    # get the date from the file
    date = Path(file).name.split(".")[1][1:]
    # get the time from the file
    time = Path(file).name.split(".")[2]
    datetime_str = f"{date}.{time}"

    return datetime_str

def format_modis_dates(time: str) -> str:
    """
    Function to format the date/time string.
    
    Args:
        time (str): The time string to be formatted.
        
    Returns:
        str: The formatted time string.
    """
    # Split the string into date and time parts
    date_str, time_str = time.split(".")
    # Convert the date part to a datetime object
    date = datetime.strptime(date_str, "%Y%j")
    # Convert the time part to a timedelta object
    time = timedelta(hours=int(time_str[:2]), minutes=int(time_str[2:]))
    # Add the date and time parts to get a datetime object
    dt = date + time
    # Convert the datetime object to a string in the format "YYYYMMDDHHMMSS"
    str_time = dt.strftime("%Y%m%d%H%M%S")

    return str_time



def _check_earthdata_login() -> bool:
    """check if earthdata login is available in environment variables"""

    if os.environ.get("EARTHDATA_USERNAME") is None or os.environ.get("EARTHDATA_PASSWORD") is None:
        msg = "Please set your Earthdata credentials as environment variables using:"
        msg += "\nexport EARTHDATA_USERNAME=<your username>"
        msg += "\nexport EARTHDATA_PASSWORD=<your password>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--earthdata-username <your username> --earthdata-password <your password>"
        raise ValueError(msg)
    
    # check if credentials are valid
    auth_obj = earthaccess.login('environment')

    if auth_obj.authenticated: 
        return True
    else:
        msg = "Earthdata login failed."
        msg += "\nPlease check your credentials and set them as environment variables using:"
        msg += "\nexport EARTHDATA_USERNAME=<your username>"
        msg += "\nexport EARTHDATA_PASSWORD=<your password>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--earthdata-username <your username> --earthdata-password <your password>"
        raise ValueError(msg)