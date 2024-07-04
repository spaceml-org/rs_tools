from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass(order=True, frozen=True)
class GOESFileName:
    """
    GOES Data product file name
    noaa-goes[sat-no]/[instrument]-[level]-[product][domain]/[year]/[day]/[hour]/
        OR_[instrument]-[level]-[product][domain]-M4C[channel]_G[sat-no]_s[YYYYDDDHHMMSSS]_e[YYYYDDDHHMMSSS]_c[YYYYDDDHHMMSSS].nc
        s = start date
        e = end date
        c = creation date
    GOES Cloud mask file name
    noaa-goes[sat-no]/[instrument]-[level]-[product][domain]/[year]/[day]/[hour]/
        OR_[instrument]-[level]-[product][domain]-M6_G[sat-no]_s[YYYYDDDHHMMSSS]_e[YYYYDDDHHMMSSS]_c[YYYYDDDHHMMSSS].nc

    """
    save_path: str 
    instrument: str
    processing_level: str
    satellite_number: str
    product: str
    domain: str
    identifier: str
    ext: str
    datetime_acquisition_start: datetime
    datetime_acquisition_end: datetime
    datetime_processing: datetime

    @classmethod
    def from_filename(cls, file_name: str):
        """
        Creates a GOESFileName object from a given file name.

        Args:
            cls (type): The class object that the method is bound to.
            file_name (str): The file name to parse.

        Returns:
            GOESFileName: The parsed GOESFileName object.
        """

        file_name = Path(file_name)
        print(file_name)
        components = file_name.name.split('_')
        save_path = str(file_name.parents[0])

        data_product = components[1]
        data_product_components = data_product.split('-')
        instrument = data_product_components[0]
        processing_level = data_product_components[1]
        product = data_product_components[2][:-1]
        domain = data_product_components[2][-1]
        identifier = data_product_components[3]
        satellite_number = components[2][-2:]

        # acquisition start time and date
        time = components[3][8:]
        year = components[3][1:5]
        day = components[3][5:8]
        datetime_acquisition_start = datetime.strptime(f"{year}{day}{time}", "%Y%j%H%M%S%f")

        # acquisition end time and date
        time = components[4][8:]
        year = components[4][1:5]
        day = components[4][5:8]
        datetime_acquisition_end = datetime.strptime(f"{year}{day}{time}", "%Y%j%H%M%S%f")

        # processing time and date
        time = components[5][8:-3]
        year = components[5][1:5]
        day = components[5][5:8]
        datetime_processing = datetime.strptime(f"{year}{day}{time}", "%Y%j%H%M%S%f")
        
        ext = components[5].split('.')[1]
        return cls(
            save_path=save_path,
            instrument = instrument,
            processing_level = processing_level,
            satellite_number=satellite_number,
            product=product,
            domain=domain,
            identifier=identifier,
            datetime_acquisition_start=datetime_acquisition_start,
            datetime_acquisition_end=datetime_acquisition_end,
            datetime_processing=datetime_processing,
            ext = ext
            )
    
    @property
    def goes_filename(self):
        """
        Generates the GOES file name based on the object's properties.

        Returns:
            str: The generated GOES file name.
        """
        # data product
        filename = f"OR_{self.instrument}-{self.processing_level}-{self.product}{self.domain}-{self.identifier}"
        # satellite number
        filename += f"_G{self.satellite_number}"
        # acquisition start and end
        date_acquisition_start = self.datetime_acquisition_start.strftime("%Y%j%H%M%S%f")
        date_acquisition_end = self.datetime_acquisition_end.strftime("%Y%j%H%M%S%f")
        filename += f"_s{date_acquisition_start}_e{date_acquisition_end}"
        # processing time
        date_processing = self.datetime_processing.strftime("%Y%j%H%M%S%f")
        filename += f"_c{date_processing}"
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
        return Path(self.save_path).joinpath(self.goes_filename)
    