# GOES16 Pipeline


> In this tutorial, we will walk through 

### Downloading Data

> In this tutorial, we will take an hours worth of data.

```bash
python rs_tools satellite=goes stage=download
```

We can see in the directory:
```bash
rs_tools/config/example/download.yaml

```

We can change the download directory.

```bash
python rs_tools satellite=goes stage=download save_dir="/path/to/new/save/dir"
```

We also allow the user to change the start and end dates.

```bash
python rs_tools satellite=goes stage=download save_dir="/path/to/new/save/dir" period.start_date="2020-10-01" period.end_date="2020-10-02" period.start_time="09:00:00" period.end_time="23:00:00" satellite.download.time_step="6:00:00"
```


```bash
python rs_tools satellite=goes stage=download save_dir="/pool/usuarios/juanjohn/data/iti/raw" period.start_date="2020-10-01" end_date="2020-10-02" period.start_time="09:00:00" period.end_time="21:00:00" time_step="12:00:00"

PYTHONPATH="." python rs_tools satellite=goes stage=geoprocess read_path="/pool/usuarios/juanjohn/data/iti/" save_path="/pool/usuarios/juanjohn/data/iti/" satellite.geoprocess.resolution=10000

PYTHONPATH="." python rs_tools satellite=goes stage=patch read_path="/pool/usuarios/juanjohn/data/iti/" save_path="/pool/usuarios/juanjohn/data/iti/" nan_cutoff=1.0 patch_size=8
```

**Warning**: 
The data is quite heavy. 
Make sure you have enough space!

### GeoProcessing Data

In this stage, we will do some light geoprocessing to "harmonize" the data.

```bash
python rs_tools satellite=goes stage=geoprocess read_path="/path/to/new/save/dir"
```


***

**TODO**: Explain the Concepts:
- Start/End Date
- Start/End Time
- Daily Window
- Time Step