import os
import earthaccess
from typing import List
from dataclasses import dataclass
from odc.geo.geom import Geometry, polygon
import geopandas as gpd
import pandas as pd
from earthaccess.results import DataGranule


@dataclass
class EAGranule:
    """
    Represents a granule of Earth observation data.

    Attributes:
        granule (DataGranule): The data granule associated with the Earth observation.
    """

    granule: DataGranule
    
    @property
    def datetime(self):
        """
        Returns the datetime of the Earth observation.

        Returns:
            datetime.datetime: The datetime of the Earth observation.
        """
        # get datetime
        datetime = self.granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
        return datetime.strptime(datetime, "%Y-%m-%dT%H:%M:%S.%fZ")
    
    @property
    def polygon(self):
        """
        Returns the polygon representing the spatial extent of the Earth observation.

        Returns:
            shapely.geometry.Polygon: The polygon representing the spatial extent.
        """
        points_list = self.granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
        coordinates = [(point['Longitude'], point['Latitude']) for point in points_list]
        return polygon(coordinates, crs="4326")
    
    @property
    def satellite_id(self):
        """
        Returns the ID of the satellite associated with the Earth observation.

        Returns:
            str: The ID of the satellite.
        """
        return self.granule["umm"]["CollectionReference"]["ShortName"]
    
    @property
    def satellite_name(self):
        """
        Returns the name of the satellite associated with the Earth observation.

        Returns:
            str: The name of the satellite.
        """
        return self.granule["umm"]["Platforms"][0]["ShortName"].lower()
    
    @property
    def filename(self):
        """
        Returns the filename of the Earth observation data.

        Returns:
            str: The filename of the Earth observation data.
        """
        return self.granule["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    
    @property
    def satellite_instrument(self) -> str:
        """
        Returns the name of the instrument used by the satellite.

        Returns:
            str: The name of the satellite instrument.
        """
        return self.granule["umm"]["Platforms"][0]["Instruments"][0]["ShortName"]
    
    @property
    def daynight_flag(self):
        """
        Returns the day/night flag of the Earth observation.

        Returns:
            str: The day/night flag of the Earth observation.
        """
        return self.granule["umm"]["DataGranule"]["DayNightFlag"]
    
    @property
    def geo_dataframe(self):
        """
        Returns a GeoDataFrame representing the Earth observation.

        Returns:
            geopandas.GeoDataFrame: A GeoDataFrame representing the Earth observation.
        """

        # create pandas geodataframe
        df = pd.DataFrame({
            "time": self.datetime,
            "satellite_id": self.satellite_id,
            "satellite_name": self.satellite_name,
            "satellite_instrument": self.satellite_instrument,
            "filename": self.filename,
            "daynight": self.daynight_flag,
            })

        # convert the list of polygons into a GeoSeries
        geometry = gpd.GeoSeries(self.polygon)

        # create a GeoDataFrame 
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        
        return gdf
        

def ea_granule_to_gdf(ea_granules: List[DataGranule]) -> gpd.GeoDataFrame:
    """
    Converts a list of EA granules to a single GeoDataFrame.

    Args:
        ea_granules (List[DataGranule]): A list of EA granules.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing all the granules.

    """
    geo_dataframes = []
    
    for igranule in ea_granules:
        
        # parse all relevant information for granule
        igranule = EAGranule(igranule)
        
        # convert to geo dataframe
        igdf = igranule.geo_dataframe
        
        # append to long list
        geo_dataframes.append(igdf)
    
    
    # concat all geo dataframes
    geo_dateframes = pd.concat(geo_dataframes, ignore_index=True)

    # clean the dataframe
    geo_dateframes = geo_dateframes.drop_duplicates().reset_index(drop=True)
    
    return geo_dataframes


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