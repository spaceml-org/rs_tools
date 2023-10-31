import shutil
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import s3fs
import os
import argparse
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

"""GOES data downloader from AWS S3 bucket inspired by goes2go from https://github.com/blaylockbk/goes2go"""

fs = s3fs.S3FileSystem(anon=True)

class GOESDownloader:
    def __init__(self, base_path):
        self.base_path = base_path
        self.dirs = ["G16", "G17", "G18"]
        #self.product = product
        [os.makedirs(os.path.join(base_path, dir), exist_ok=True) for dir in self.dirs]
        #os.makedirs(os.path.join(base_path, self.satellite, self.product), exist_ok=True)

    def download(self, df, base_path):
        """Download the files from a DataFrame listing with multithreading"""

        for i in tqdm(range(len(df))):
            #file_path = os.path.join(base_path, df.satellite[i], df.product_mode[i], "%s.nc" % df.start[i].isoformat("T", timespec='seconds'))
            fs.get(df.file[i], os.path.join(base_path, df.satellite[i], df.product_mode[i], "%s.nc" % df.start[i].isoformat("T", timespec='seconds')))

    def goes_file_df(self, satellite, product, start, end, bands=None, refresh=True):

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        DATES = pd.date_range(f"{start:%Y-%m-%d %H:00}", f"{end:%Y-%m-%d %H:00}", freq="1H")
        files = []
        for DATE in DATES:
            files += fs.ls(f"{satellite}/{product}/{DATE:%Y/%j/%H/}", refresh=refresh)

        # Build a table of the files
        # --------------------------
        df = pd.DataFrame(files, columns=["file"])
        df[["product_mode", "satellite", "start", "end", "creation"]] = (
        df["file"].str.rsplit("_", expand=True, n=5).loc[:, 1:]
            )

        if product.startswith("ABI"):
           product_mode = df.product_mode.str.rsplit("-", n=1, expand=True)
           df["product"] = product_mode[0]
           df["mode_bands"] = product_mode[1]

           mode_bands = df.mode_bands.str.split("C", expand=True)
           df["mode"] = mode_bands[0].str[1:].astype(int)
           try:
               df["band"] = mode_bands[1].astype(int)
           except:
               # No channel data
               df["band"] = None

           # Filter files by band number
           if bands is not None:
               if not hasattr(bands, "__len__"):
                   bands = [bands]
               df = df.loc[df.band.isin(bands)]

        df["start"] = pd.to_datetime(df.start, format="s%Y%j%H%M%S%f")
        df["end"] = pd.to_datetime(df.end, format="e%Y%j%H%M%S%f")
        df["creation"] = pd.to_datetime(df.creation, format="c%Y%j%H%M%S%f.nc")

        #Filter files by requested time range
        df = df.loc[df.start >= start].loc[df.end <= end].reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download GOES data')
    parser.add_argument('--download_dir', type=str, help='path to the download directory.')
    parser.add_argument('--n_workers', type=str, help='number of parallel threads.', required=False, default=4)

    args = parser.parse_args()
    # base_path = args.download_dir
    # n_workers = args.n_workers
    # satellite = args.satellite
    # product = args.product
    base_path = '/Users/christophschirninger/PycharmProjects/MDRAIT_ITI/GOES'

    download_util = GOESDownloader(base_path=base_path)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 1, 2)
    product = ["ABI-L2-ACHAF", "ABI-L2-ACHTF", "ABI-L2-ACMF"]

    dataframe = download_util.goes_file_df(satellite="noaa-goes16", product=product[0], start=start_date, end=end_date, bands=None,
                             refresh=True)
    download_util.download(dataframe, base_path)