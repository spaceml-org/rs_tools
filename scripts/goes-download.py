"""


Summary: The download script to interact directly with the goes2go package.
We only want to specify what is necessary and compatible with the goes2go package.
In general, we want to satellite, the spatial domain, and the period.

Args:
    satellite_number: 

**Input Parameters**

Downloading:
- satellite number: int --> (16,17,18)
- spatial extent: str --> full disk (F), CONUS (C), Mesoscale domains (M, M1, M2)
- goes instrument: str --> e.g. ABI radiance (or SUVI for helio)
- preprocessing level: str --> e.g. level-1b
- directory: str
- return xarray dataset or list of files --> return as file list works better?
- band specifications: list[int] --> download all or subset only
- start time (of range of times to be downloaded): str
- end time: str
- timesteps/number of files: str
- day vs. night mode: --> e.g. for only downloading day mode images

Basic Processing:
- resolution: --> downscale all bands to common resolution (e.g. 2 km)
- coordinate system transformations
- etc.


# download stuff
np.arange, np.linspace
t0, t1, dt | num_files
timestamps = [t0, t1, t2]

for itime in timestamps:
    # download to folder
    # open data in folder
    # preprocess - downscale, aggregate bands, coordinate reference system transformations
    # resave

# preprocess stuff
preprocess_stuff(*args, **kwargs)

########## GOES ############
# ==================
# DOWNLOAD
# ==================
# download data
download_data(*arg, **kwargs)

# ==================
# QUALITY CHECKS
# ==================
# day & night flag
ds: xr.Dataset = day_and_night_flag(ds, *args, **kwargs)
# all bands are there?
ds: xr.Dataset = check_all_bands(ds, *args, **kwargs)

# ==================
# PREPROCESSING
# ==================
# open dataset
ds: xr.Dataset = ...
# crs transformation
ds: xr.Dataset = crs_transform(ds, *args, **kwargs)
# upscale/downscale
ds: xr.Dataset = resample(ds, *args, **kwargs)


"""