# MODIS


Below are some notes on NASA's Terra and Aqua satellites, specifically focussing on the [Moderate Resolution Imaging Spectrometer](https://modis.gsfc.nasa.gov/about/) (MODIS).

## Terra & Aqua Satellites

> The [Terra](https://terra.nasa.gov/) and [Aqua](https://aqua.nasa.gov/) satellites, launched in 1999 and 2002 respectively, are cornerstones in NASA's Earth Observation Program, and have collected invaluable measurements of Earth's land, oceans, cryosphere and atmosphere over the last two decades. 
> Both satellites follow a near-polar, Sun-synchronous orbit, i.e. they pass over the same point on Earth at the same time each day. Terra orbits the Earth as part of the Morning Train, while Aqua is part of the [Afternoon Train](https://en.wikipedia.org/wiki/A-train_(satellite_constellation)) of satellites. Both satellites image the same part of Earth approximately 3 hours apart, and together provide global coverage every 1 - 2 days. The orbit repeat cycle for each satellite is 16 days.

### Terra Instruments
* [Advanced Spaceborne Thermal Emission and Reflection Radiometer](https://terra.nasa.gov/about/terra-instruments/aster) (ASTER)
* [Clouds and the Earth’s Radiant Energy System](https://terra.nasa.gov/about/terra-instruments/ceres) (CERES)
* [Multi-angle Imaging SpectroRadiometer](https://terra.nasa.gov/about/terra-instruments/misr) (MISR)
* [Moderate Resolution Imaging Spectrometer](https://terra.nasa.gov/about/terra-instruments/modis) (MODIS)
* [Measurement of Pollution in the Troposphere](https://terra.nasa.gov/about/terra-instruments/mopitt) (MOPITT)

### Aqua Instruments
* [Atmospheric Infrared Sounder](https://aqua.nasa.gov/content/airs) (AIRS)
* [Advanced Microwave Sounding Unit ](https://aqua.nasa.gov/content/amsu) (AMS-U)
* [Humidity Sounder for Brazil](https://aqua.nasa.gov/content/hsb) (HSB)
* [Advanced Microwave Scanning Radiometer for EOS](https://aqua.nasa.gov/content/amsr-e) (AMSR-E)
* [Moderate Resolution Imaging Spectrometer](https://aqua.nasa.gov/modis) (MODIS)
* [Cloud's and the Earth's Radiant Energy System](https://aqua.nasa.gov/ceres) (CERES)

## MODIS Data
### Spectral Bands & Resolution

Both satellites share the MODIS instrument, which collects data in 36 spectral channels:

<img width="261" alt="Screenshot 2023-11-30 at 15 03 17" src="https://github.com/spaceml-org/rs_tools/assets/32869829/e4e3a294-a68f-4314-9396-d446ffe18231">

> Image Credit: Wikipedia

### Processing Levels

* Level-0: Raw instrument measurements (swath product)
* Level-1A: Scans of raw radiances in counts (swath product)
* Level-1B: Calibrated radiances at 250, 500, 1000 m resolution (swath product)
* Level-2: Derived geophysical variables at the same resolution and location as Level-1 source data (swath products)
* Level-2G: Level-2 data mapped on a uniform space-time grid scale (Sinusoidal)
* Level-3: Gridded variables in derived spatial and/or temporal resolutions
* Level-4: Model output or results from analyses of lower-level data

In this project, we mainly work with Level 1-B data.

### Naming Conventions

> The MODIS Level-1 data files are named according to the following [naming conventions](https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/modis-overview/#modis-naming-conventions): **SATXX.AYYYYDD.HHDD.CCC.YYYYDDDHHMMSS.hdf**

* SAT: Satellite (Terra -> MOD, Aqua -> MYD, Combined Product -> MCD)
* XX: Other product details (e.g. QKM -> 250 m, HKM -> 500 m, 1KM -> 1000 m resolution)
* AYYYYDD: Julian Day of Acquisition
* HHDD: Time of Acquisition
* CCC: Collection
* YYYYDDDHHMMSS: Julian Day of Production

### Day & Night Mode

> MODIS continuously collects measurements in either **day mode**, **night mode**, or **mixed mode**, depending on the time of day of the are under observation. During each orbit, 9 **day mode**, 9 **night mode**, and 2 **mixed mode** granules are measured respectively.
> To reduce storage space and transmission of files containing no useful data, Level-1B allows writing of 250 m and 500 m data files to be turned off for granules that contain no day mode scans. When data are transmitted in night mode, the Reflective Solar Bands (bands 1-19) are empty and appear to contain fill values of "65535".
> While **day** and **mixed mode** files are usually 200-300 MB large, **night mode** files are often <100 MB.

## Working with Level-1B Data

A jupyter notebook on how to download, open, and plot Level-1B data is provided in the main repository. Below, we summarise some of the key takeaways from working with Level-1B data. 

Each Level-1B granule contains multiple data variables, including (1) the science data on a pixel-to-pixel basis, (2) uncertainty information on a pixel-to-pixel basis, (3) geolocation data for the pixels, and (4) metadata. Exactly how many variables are contained in each file depends on the spatial resolution. The 1KM aggregated data, for instance, contains 27 data variables, for which the science data is stored in the following variable names:

**EV_250_Aggr1km_RefSB:** Earth View 250M Aggregated 1km Reflective Solar Bands Scaled Integers
* Bands: 1, 2

**EV_500_Aggr1km_RefSB:** Earth View 500M Aggregated 1km Reflective Solar Bands Scaled Integers
* Bands: 3, 4, 5, 6, 7

**EV_1KM_RefS:** Earth View 1KM Reflective Solar Bands Scaled Integers
* Bands: 8, 9, 10, 11, 12, 13lo, 13hi, 14lo, 14hi, 15, 16, 17, 18, 19, 26

**EV_1KM_Emissive:** Earth View 1KM Emissive Bands Scaled Integers
* Bands: 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36

Note that the data is provided in "scaled integers" and does not yet include units. To convert the scaled integers into radiances (or reflectances), the data needs to be corrected by subtracting the radiance_offsets (reflectance_offsets) and multiplying with the radiance_scales (reflectance_scales). After this correction step, radiances are provided in $W/m^2/\micro m/sr$, while reflectance is still unit-less. More information on this correction step is provided in the [MODIS Level-1B User Guide](https://ccplot.org/pub/resources/Aqua/MODIS%20Level%201B%20Product%20User%20Guide.pdf).

In addition to the science data, geolocation information is provided for each granule. Note that the size of the latitude/longitude coordinates (406, 271) does not match the size of the data (2030, 1354). This is because geolocation data is only provided for a subset of the pixels. To match the geolocation data to the size of the data, the provided latitude/longitude coordinates can be interpolated.

## Level 2: Cloud Mask (35_L2)

The [MODIS Cloud Mask product](https://atmosphere-imager.gsfc.nasa.gov/products/cloud-mask) is a Level 2 product generated at 1-km and 250-m (at nadir) spatial resolutions from MODIS visible, near-infrared and infrared bands to automatically assign one of 4 classes to each pixel:

- cloudy
- probably cloudy
- probably clear
- clear

There are two MODIS Cloud Mask data product files: MOD35_L2 (for Terra satellite) and MYD35_L2 (for Aqua satellite).


## Data Format & Access

MODIS data is provided in the [.hdf file format](https://asdc.larc.nasa.gov/documents/tools/hdf.pdf).

MODIS data can be downloaded via NASA's EarthData graphical user interface ([https://ladsweb.modaps.eosdis.nasa.gov/search/](https://ladsweb.modaps.eosdis.nasa.gov/search/)) and data archive ([https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61)), or via the USGS ([https://e4ftl01.cr.usgs.gov/](https://e4ftl01.cr.usgs.gov/)). Note that the latter brings you directly to a US government computer that hosts the MODIS data (MOLA --> AQUA data, MOLT --> TERRA data, MOTA --> AQUA & TERRA data); the terms and conditions of usage should be respected when accessing this resource. As part of rs_tools, we developed easy-to-use download scripts to download data directly from NASA EarthData.

## Software Tools

**Google Earth Engine**.

* [gee-tool](https://github.com/gee-community/gee_tools)
* [wxee](https://wxee.readthedocs.io/en/latest/) - xarray-based

**Other Tools**

* [modis-tool](https://github.com/fraymio/modis-tools)
* [pansat](https://github.com/SEE-GEO/pansat)
* [py-modis](http://www.pymodis.org/)

---
## Q/A

**Noah**

* How long it takes to download an image or set of images this way?
* Is there a "cutoff" date for when the images aren't available? --> Still collecting data today.
* Can you pre-filter based on cloud coverage? --> I don't think so?