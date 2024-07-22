from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import pandas as pd
import os
import eumdac
from eumdac.product import Product, ProductError
from loguru import logger
import shutil


@dataclass(order=True, frozen=True)
class MSGFileName:
    """
    MSG Data product file name
    MSG[sat-no]-[instrument]-MSG[data-product]-0100-NA-[YYYYMMDDHHMMSS].[sssssssss]Z-NA.nat

    MSG Cloud mask file name
    MSG[sat-no]-[instrument]-MSG[data-product]-0100-0100-[YYYYMMDDHHMMSS].[sssssssss]Z-NA.grb
    """
    save_path: str 
    instrument: str
    satellite_number: str
    data_product: str
    data_details_1: str
    data_details_2: str
    subsecond: str
    ext: str
    datetime_acquisition: datetime

    @classmethod
    def from_filename(cls, file_name: str):
        """
        Creates a MSGFileName object from a given file name.

        Args:
            cls (type): The class object that the method is bound to.
            file_name (str): The file name to parse.

        Returns:
            MSGFileName: The parsed GOESFileName object.
        """

        file_name = Path(file_name)
        components = file_name.name.split('-')
        save_path = str(file_name.parents[0])

        satellite_number = components[0][3:]
        instrument = components[1]
        data_product = components[2][3:]
        data_details_1 = components[3]
        data_details_2 = components[4]
        datetime_acquisition = components[5]
        ext = components[6].split('.')[1]

        # acquisition time and date
        date_time_components = datetime_acquisition.split('.')
        year = date_time_components[0][:4]
        month = date_time_components[0][4:6]
        day = date_time_components[0][6:8]
        time = date_time_components[0][8:14]
        subsecond = date_time_components[1][:-1]
        datetime_acquisition = datetime.strptime(f"{year}{month}{day}{time}", "%Y%m%d%H%M%S")

        ext = components[6].split('.')[1]
        return cls(
            save_path=save_path,
            instrument = instrument,
            satellite_number=satellite_number,
            data_product=data_product,
            data_details_1=data_details_1,
            data_details_2=data_details_2,
            subsecond=subsecond,
            ext = ext,
            datetime_acquisition=datetime_acquisition
            )
    
    @property
    def msg_filename(self):
        """
        Generates the MSG file name based on the object's properties.

        Returns:
            str: The generated MSG file name.
        MSG[sat-no]-[instrument]-MSG[data-product]-0100-NA-[YYYYMMDDHHMMSS].[sssssssss]Z-NA.nat

        """
        # satellite number
        filename = f"MSG{self.satellite_number}"
        # instrument
        filename += f"-{self.instrument}"
        # data product
        filename += f"-MSG{self.data_product}-{self.data_details_1}-{self.data_details_2}"
        # acquisition time
        date_acquisition = self.datetime_acquisition.strftime("%Y%m%d%H%M%S")
        filename += f"-{date_acquisition}"
        # subsecond
        filename += f".{self.subsecond}Z-NA"
        # extension
        filename += f".{self.ext}"
        return filename
    
    @property
    def full_path(self):
        """
        Gets the full path of the GOES file.

        Returns:
            Path: The full path of the GOES file.
        """
        return Path(self.save_path).joinpath(self.msg_filename)


@dataclass
class MSGGranule:
    """
    Represents a granule of MSG data.
    
    MSG[sat-no]-[instrument]-MSG[data-product]-0100-NA-[YYYYMMDDHHMMSS].[sssssssss]Z-NA

    The MSGGranule class provides methods and properties to access and manipulate
    metadata associated with a granule of MSG data.

    Attributes:
        granule (Product): The product associated with the granule.

    Properties:
        title (str): The title of the granule.
        product_id (str): The product ID of the granule.
        date (pandas.Timestamp): The date of the granule.
        ext (str): The file extension of the granule.

    Methods:
        __post_init__(): Initializes the MSGGranule object.
    """
    granule: Product
    
    def __post_init__(self):
        components = self.product_id.split("-")
        self.satellite_number = components[0][3:]
        self.instrument = components[1]
        self.data_product = components[2][3:]
        self.data_details_1 = components[3]
        self.data_details_2 = components[4]
        datetime_acquisition = components[5]

        # acquisition time and date
        date_time_components = datetime_acquisition.split('.')
        year = date_time_components[0][:4]
        month = date_time_components[0][4:6]
        day = date_time_components[0][6:8]
        time = date_time_components[0][8:14]
        self.subsecond = date_time_components[1][:-1]
        self.datetime_acquisition = pd.to_datetime(f"{year}{month}{day}{time}", format="%Y%m%d%H%M%S")
    
    def __str__(self):
        return (
            "MSG Granule:"
            + f"\nCollection ID: {self.collection_id}"
            + f"\nProduct ID: {self.product_id}"
            + f"\nTimeStamp: {self.datetime_acquisition}"
            + f"\nExtension: {self.ext}"
            + f"\nFileName: {self.filename}"
        )
        
    def __repr__(self):
        return self.__str__()
    @property
    def product_id(self):
        return self.granule.metadata["download_properties"]["identifier"]
    
    @property
    def collection_id(self):
        return self.granule.metadata["download_properties"]["parentIdentifier"]
    
    @property
    def timestamp(self):
        return pd.to_datetime(self.granule.metadata["properties"]["date"].split("/")[0])
    
    @property
    def ext(self):
        return MSG_FORMAT[self.granule.metadata["download_properties"]["productInformation"]["processingInformation"]["format"]]
    
    @property
    def filename(self):
        return f"{self.product_id}{self.ext}"
    
    def download(self, save_dir: str):
        try:
            with self.granule.open(entry=self.filename) as fsrc:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path.joinpath(fsrc.name)
                with open(save_path, mode="wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
        except ProductError as error:
            msg = f"Could not download {self.filename} from '{self.granule}': "
            msg += f"{error.msg}"
            print(msg)
        
        
def granule_to_dataframe(
    granule: MSGGranule,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "collection_id": granule.collection_id,
            "product_id": granule.product_id,
            "time": granule.datetime_acquisition,
            "ext": granule.ext,
            "filename": granule.filename
            
        },
        index=[0]
    )
    return df
            
            

MSG_FORMAT = {
    "GRIB2": ".grb",
    "MSGNative": ".nat"
}

    
def _check_eumdac_login(eumdac_key: str, eumdac_secret: str) -> bool:
    """check if eumdac login is available in environment variables / as input arguments"""
    if eumdac_key and eumdac_key:
        os.environ["EUMDAC_KEY"] = eumdac_key
        os.environ["EUMDAC_SECRET"] = eumdac_secret

    if os.environ.get("EUMDAC_KEY") is None or os.environ.get("EUMDAC_SECRET") is None:
        msg = "Please set your EUMDAC credentials as environment variables using:"
        msg += "\nexport EUMDAC_KEY=<your user key>"
        msg += "\nexport EUMDAC_SECRET=<your user secret>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--eumdac-key <your user key> --eumdac-secret <your user secret>"
        raise ValueError(msg)
    else:
        eumdac_key = os.environ.get("EUMDAC_KEY")
        eumdac_secret = os.environ.get("EUMDAC_SECRET")
        
    # check if credentials are valid
    credentials = (eumdac_key, eumdac_secret)
    try:
        token = eumdac.AccessToken(credentials)
        # 
        return token
    except:
        msg = "EUMDAC login failed."
        msg += "\nPlease check your credentials and set them as environment variables using:"
        msg += "\nexport EUMDAC_KEY=<your user key>"
        msg += "\nexport EUMDAC_SECRET=<your user secret>"
        msg += "\nOr provide them as command line arguments using:"
        msg += "\n--eumdac-key <your user key> --eumdac-secret <your user secret>"
        logger.debug(f"EUMDAC login unsuccessful. Token '{token}' expires {token.expiration}")
        raise ValueError(msg)    

