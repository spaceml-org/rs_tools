from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
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
        print(file_name)
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