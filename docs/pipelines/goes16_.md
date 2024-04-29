# GOES16 Pipeline




## Downloading Data

Firstly, we will download some data. The default configuration will consist of downloading over a years worth of data.

```bash
python rs_tools satellite=goes stage=download
```


---
### Configuration

We can peek into the `rs_tools/config/example/download.yaml` configuration file to see some of the options we have to modify this.


```yaml
# PERIOD
period:
  start_date: '2020-10-01'
  start_time: '00:00:00'
  end_date: '2020-10-31'
  end_time: '23:59:00'

# CLOUD MASK
cloud_mask: True
  
# PATH FOR SAVING DATA
save_dir: data

defaults:
  - _self_
```

We see that we can change some of the configurations available. 
For this example, we will change this to download only a subset of data.

We also have some more general options for the user to change which are satellite specific.
These can be found in `rs_tools/config/example/satellite/goes.yaml`

```yaml
download:
  _target_: rs_tools._src.data.goes.downloader_goes16.download
  save_dir: ${save_dir}/goes16/raw
  start_date: ${period.start_date}
  start_time: ${period.start_time}
  end_date: ${period.end_date}
  end_time: ${period.end_time}
  daily_window_t0: "14:00:00"
  daily_window_t1: "20:00:00"
  time_step: "1:00:00"

```

We will change the save directory, start/end time, and the time step.

```bash
python rs_tools satellite=goes stage=download save_dir="/path/to/savedir" period.start_date="2020-10-01" period.end_date="2020-10-02" period.start_time="09:00:00" period.end_time="21:00:00" satellite.download.time_step="6:00:00"
```

We notice that there are some files that should be available for processing. 
In particular, we have two sets of files: the Level 1 Radiances and the Cloud Mask.


```bash
/path/to/savedir/goes16/raw/goes16/L1b
/path/to/savedir/goes16/raw/goes16/CM
```

We use both in this project.


---
## Geoprocessing


We have an extensive geoprocessing steps to be able to 

We can peek into the `rs_tools/config/example/download.yaml` configuration file to see some of the options we have to modify this.


```yaml
# PERIOD
geoprocess:
  _target_: rs_tools._src.geoprocessing.goes.geoprocessor_goes16.geoprocess
  read_path: ${read_path}/goes16/raw
  save_path: ${save_path}/goes16/geoprocessed
  resolution: null
  region: "-130 -15 -90 5"
  resample_method: bilinear
```

The most important options are the `resolution` and the `region`.
The resolution is a float or integer that is measured in km.

Below, we have an example of the command we 


```bash
python rs_tools satellite=goes stage=download save_dir="/path/to/new/save/dir" period.start_date="2020-10-01" period.end_date="2020-10-02" period.start_time="09:00:00" period.end_time="23:00:00" satellite.time_step="6:00:00"
```


We can see the saved data are clean

```bash
/path/to/savedir/goes16/geoprocessed/20201001150019_goes16.nc
/path/to/savedir/goes16/geoprocessed/20201002150019_goes16.nc
```

```bash
PYTHONPATH="." python rs_tools satellite=goes stage=download save_dir="/pool/usuarios/juanjohn/data/iti/raw" period.start_date="2020-10-01" period.end_date="2020-10-02" period.start_time="09:00:00" period.end_time="21:00:00" satellite.download.time_step="6:00:00"

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