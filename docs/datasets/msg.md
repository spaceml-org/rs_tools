# MSG

Below are some notes on the Meteosat Second Generation (MSG) Spinning Enhanced Visible and Infrared Imager (SEVIRI) instrument.

## MSG Satellites

The MSG satellite family consists of four operational meteorological satellites in geostationary orbit: Meteosat-8, Meteosat-9, Meteosat-10, and Meteosat-11. Information about launch dates and operational periods is given below. All satellites have a primary field of view over Europe. After their main operational period, Meteosat-8 and Meteosat-10 were relocated to perform measurements over the Indian Ocean. During this time, they are referred to as Meteosat-8 (IODC) and Meteosat-10 (IODC). 
MSG superseded the Meteosat First Generation (MFG) and is followed by the Metesat Third Generation (MTG). All satellites contain the same instruments, some of which were also flown on other Meteosat satellites.

Meteosat-8:
* Launch Date: 28 Aug 2002
* End of Life: 04 Jul 2016
* Position: 3.7° E

Meteosat-8 (IODC):
* Launch Date: 15 Sep 2016
* End of Life: Nov 2022
* Position: 41.5° E

Meteosat-9:
* Launch Date: 21 Dec 2005
* End of Life: 01 Apr 2022
* Position: 3.5° E

Meteosat-9 (IODC):
* Launch Date: 01 Jul 2022
* End of Life: Operational (as of 01/2024)
* Position: 45.5° E

Meteosat-10:
* Launch Date: 05 Jul 2012
* End of Life: Operational (as of 01/2024)
* Position: 0.0° E

Meteosat-11:
* Launch Date: 15 Jul 2015
* End of Life: Operational (as of 01/2024)
* Position: 9.5° E

### Instruments:
* [Data Collection Service](https://space.oscar.wmo.int/instruments/view/dcs_meteosat) (DCS)
* [Geostationary Search and Rescue](https://space.oscar.wmo.int/instruments/view/geos_r) (GOES&R)
* [Geostationary Earth Radiation Budget](https://space.oscar.wmo.int/instruments/view/gerb) (GERB)
* [Spinning Enhanced Visible and Infrared Imager](https://space.oscar.wmo.int/instruments/view/seviri) (SEVIRI)

## SEVIRI Data
### Spectral Bands & Resolution

The SEVIRI instruments measure 12 spectral channels.

![MSG-SEVIRI-spectral-channels-VIS-visible-IR-infrared-WV-water-vapour](https://github.com/spaceml-org/rs_tools/assets/32869829/9c61f782-e955-4827-bb64-c76f8221af55)

> Image Credit: Vázquez-Navarro et al. A fast method for the retrieval of integrated longwave and shortwave top-of-atmosphere irradiances from MSG/SEVIRI (RRUMS), Atmospheric Measurement Techniques 5(4):4969-5008 (2012), DOI: [10.5194/amtd-5-4969-2012](http://dx.doi.org/10.5194/amtd-5-4969-2012) 

The resolution of the instrument is 1 km at the sub-satellite point (SSP) for the high resolution visible (HRV) channel, and 3 km for all other channels. The resolution decreases closer to the image limb.

![Spatial-resolution-and-coverage-of-the-SEVIRI-instrument-33](https://github.com/spaceml-org/rs_tools/assets/32869829/342dbb99-72e0-4acb-83b3-b3caff63e87c)

> Image Credit: Eissa et al. Validation of the Surface Downwelling Solar Irradiance Estimates of the HelioClim-3 Database in Egypt, Remote Sens. 2015, 7(7), 9269-9291, DOI: [10.3390/rs70709269](http://dx.doi.org/10.3390/rs70709269)

### Instrument Field-of-view

An example field-of-view (FOV) the satellites positioned at 0.0° E  and over the Indian Ocean are shown below.

<img width="1007" alt="Screenshot 2024-02-01 at 17 43 03" src="https://github.com/spaceml-org/rs_tools/assets/32869829/444fc468-5b40-42f6-9cfb-7e71f4278147">

<img width="1005" alt="Screenshot 2024-02-01 at 17 42 49" src="https://github.com/spaceml-org/rs_tools/assets/32869829/c88e9981-128a-4563-aa47-5c684570bce0">

> Credit: [EUMETSAT](https://navigator.eumetsat.int/)

In addition to the full disk measurements, the rapid scan mode is performed over the following FOV.

<img width="1011" alt="Screenshot 2024-02-01 at 17 44 56" src="https://github.com/spaceml-org/rs_tools/assets/32869829/a371d86f-7d07-466d-9810-dc6f43000a16">

> Credit: [EUMETSAT](https://navigator.eumetsat.int/)

### Processing Levels

* Level-0: Raw instrument measurements.
* Level-1.5: Geolocated and radiometrically pre-processed image data, ready for further processing. Spacecraft specific effects have been removed. More information about Level-1.5 data can be found in the [technical user guide](https://www-cdn.eumetsat.int/files/2020-05/pdf_ten_05105_msg_img_data.pdf).

### Naming Conventions

> The MSG Level-1 data files are named according to the following naming conventions: MSG#-IIII-MSGXX-0100-NA-YYYYMMDDHHMMSS.ssssZ-NA.nat, for example _MSG4-SEVI-MSG15-0100-NA-20211110081242.766000000Z-NA.nat_.

* #: Satellite number (e.g. 1 -> Meteosat-8, 2 -> Meteosat-9, etc.)
* IIII: Instrument details (e.g. SEVI for SEVIRI)
* XX-0100: likely referring to specific product or data types
* NA: not locationally constrained
* YYYYMMDD: Acquisition date
* HHMMSS: Acquisition time
* ssss: likely referring to sub-second acquisition time
* Z: Zulu / UTC time zone
* NA: not locationally constrained

### Measurement Frequencies

Full disk images are measured every 15 mins.

### Day & Night Mode

### Cloud Mask

Cloud masks are provided (e.g. as data product EO:EUM:DAT:MSG:CLM), and contain 4 types of pixel classifications:
* 0 = clear sky over land
* 1 = clear sky over water
* 2 = clouds
* 3 = not processed

![40c4488a-b6ea-4904-9d32-ea77992538e5](https://github.com/spaceml-org/rs_tools/assets/32869829/74f606bd-53cb-48cf-ad3a-88ceb37e8158)

## Working with Level-1 Data

MSG/SEVIRI radiances are provided in $mW/m^2/sr/cm^{-1}$, i.e. the data is normalised to wavenumbers. In order to convert the data to $W/m^2/sr/um$, the data needs to be multiplied by $10^{-7}$; $mW = 10^{-3} W$, $cm^{-1} = 10^4 {um}$.

## Data Format & Access

SEVIRI data can be downloaded via EUMETSAT's Data Access portal. The following data products are available:

* EO:EUM:DAT:MSG:CLM - Cloud Mask (0 degree position)
* EO:EUM:DAT:MSG:CLM-IODC - Cloud Mask (Indian Ocean)
* EO:EUM:DAT:MSG:RSS-CLM - Rapid Scan Cloud Mask

* EO:EUM:DAT:MSG:HRSEVIRI - High Rate SEVIRI Level 1.5 Image Data (0 degree position)
* EO:EUM:DAT:MSG:HRSEVIRI-IODC - High Rate SEVIRI Level 1.5 Image Data (Indian Ocean)
* EO:EUM:DAT:MSG:MSG15-RSS - Rapid Scan High Rate SEVIRI Level 1.5 Image Data

The data is provided in .nat format. Each file contains data of shape (3712, 3712).

## Software Tools