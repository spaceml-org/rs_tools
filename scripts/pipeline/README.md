# Demo Pipeline

**Downloading**

```python
python scripts/pipeline/download_modis.py --start-date 2018-10-01 --end-date 2018-10-05 --save-path /home/juanjohn/data/rs/modis/raw
```

**Preprocessing**

```python
python scripts/pipeline/preprocess_modis.py --read-path "/home/juanjohn/data/rs/modis/raw/modis" --save-path "/home/juanjohn/data/rs/modis/analysis"
```


```python
python scripts/pipeline/prepatch.py --read-path "/home/juanjohn/data/rs/modis/analysis"  --save-path "/home/juanjohn/data/rs/modis/mlready"
```