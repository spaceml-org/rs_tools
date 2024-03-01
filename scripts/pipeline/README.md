# Demo Pipeline

### MODIS PIPELINE

**Downloading**

```bash
python scripts/pipeline/download_modis.py --start-date 2018-10-01 --end-date 2018-10-05 --save-path /home/juanjohn/data/rs/modis/raw
```

**Preprocessing**

```bash
python scripts/pipeline/preprocess_modis.py --read-path "/home/juanjohn/data/rs/modis/raw/modis" --save-path "/home/juanjohn/data/rs/modis/analysis"
```

**Pre-Patching**

```bash
python scripts/pipeline/prepatch.py --read-path "/home/juanjohn/data/rs/modis/analysis"  --save-path "/home/juanjohn/data/rs/modis/mlready"
```


### GOES PIPELINE

**Downloading**

```bash
python scripts/pipeline/download_goes.py --save-path /home/juanjohn/data/rs/goes
```

**Preprocessing**

```bash
python scripts/pipeline/preprocess_goes.py --read-path "/home/juanjohn/data/rs/goes16/raw/noaa-goes16/" --save-path "/home/juanjohn/data/rs/goes16/analysis"
```


**Pre-Patching**