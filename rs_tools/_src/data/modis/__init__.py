import os
from typing import List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import earthaccess

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