# GOES 16

Below are some notes on NOAA's GOES satellites, specifically focussing on the [Advanced Baseline Imager](https://www.goes-r.gov/spacesegment/abi.html) (ABI).


## GOES Satellites

### GOES-16
- Launched on 19 November 2016, operational since 18 December 2017
- Longitude central point: -75.2

### GOES-17 [no longer operational]
- Launched on 1 March 2018, operational from 12 February 2019 to 4 January 2023 at longitude -136.9
- Replaced by GOES-18 due to issues with its Advanced Baseline Imager (ABI) instrument
- Moved to longitude -104.7 (between GOES-16 and GOES-18) and serves as backup for the operational satellites

### GOES-18
- Launched on 1 March 2022, operational since 4 January 2023 (replaced GOES-17)
- Longitude central point: -136.9

## [GOES Instruments](https://www.goes-r.gov/spacesegment/instruments.html)

Earth-facing:
- Advanced Baseline Imager (ABI)
- Geostationary Lightning Mapper (GLM)]

Sun-facing:
- Extreme Ultraviolet and X-ray Irradiance Sensors (EXIS)
- Solar Ultraviolet Imager (SUVI)

Space environment:
- Magnetometer (MAG)
- Space Environment In-Situ Suite (SEISS)

## ABI Data

### Processing Levels

* Level-0: Raw instrument measurements
* Level-1B: Calibrated and geolocated radiances
* Level-2: Derived geophysical variables
* Level-3: Geophysical variables mapped on uniform space-time grid

### Level-1B: Spectral Bands & Resolution

<img width="617" alt="GOES-ABI-bands" src="https://github.com/spaceml-org/rs_tools/assets/33373979/93d673d1-2eca-4a4b-84a9-d9135e7d0dd7">

### Level 2: Clear Sky Mask (ACM)
The [clear sky mask algorithm](https://www.star.nesdis.noaa.gov/goesr/documents/ATBDs/Enterprise/ATBD_Enterprise_Cloud_Mask_v1.2_2020_10_01.pdf) uses the GOES ABI visible, near-infrared and infrared bands to automatically assign one of the following 4 classes to each pixel:
- cloudy
- probably cloudy
- probably clear
- clear

ACM data is provided at the native 2km resolution on the ABI fixed grid for full disk, CONUS, and mesoscale coverage regions, at the same temporal resolution as ABI L1b data.

### Naming Conventions
GOES ABI Level 1b and 2 data are named according to the following [naming conventions](https://cimss.ssec.wisc.edu/goes/ABI_File_Naming_Conventions.pdf):

`\<SE\>\_\<DSN\>\_\<PID\>\_\<Obs Start Date & Time\>\_\<Obs End Date & Time\>\_\<Creation Date & Time\>.\<FE\>`

where:
- SE = System Environment
- DSN = Data Short Name
- PID = Platform Identifier
- Obs Start Date & Time = Observation Period Start Date & Time
- Obs End Date & Time = Observation Period End Date & Time
- Creation Date & Time = File Creation Date & Time
- FE = File Extension

## Working with Level-1B Data

GOES/ABI radiances are provided in $mW/m^2/sr/cm^{-1}$, i.e. the data is normalised to wavenumbers. In order to convert the data to $W/m^2/sr/um$, the data needs to be multiplied by $10^{-7}$; $mW = 10^{-3} W$, $cm^{-1} = 10^4 {um}$.

## Data Format & Access
GOES Data can be explored in the following buckets:

AWS:
- [GOES-16 AWS S3 Explorer](https://noaa-goes16.s3.amazonaws.com/index.html)
- [GOES-17 AWS S3 Explorer](https://noaa-goes17.s3.amazonaws.com/index.html)
- [GOES-18 AWS S3 Explorer](https://noaa-goes18.s3.amazonaws.com/index.html)

Google Cloud:
- [GOES-16 Google Cloud Bucket Explorer](https://console.cloud.google.com/storage/browser/gcp-public-data-goes-16)
- [GOES-17 Google Cloud Bucket Explorer](https://console.cloud.google.com/storage/browser/gcp-public-data-goes-17)
- [GOES-18 Google Cloud Bucket Explorer](https://console.cloud.google.com/storage/browser/gcp-public-data-goes-18) 

## Software Tools
> [GOES2GO](https://blaylockbk.github.io/goes2go/_build/html/index.html) - Software download 
* allows downloading of GOES data from AWS



## Q/A