import os
import multiprocessing
import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm
import pandas as pd

import autoroot
from rs_tools._src.utils.io import get_list_filenames

def _check_input_filetype(file_type: str) -> bool:
    """checks allowed input file types."""
    if file_type in ["nc"]:
        return True
    else:
        msg = "Unrecognized file type"
        msg += f"\nNeeds to be 'nc'. Others are not yet implemented."
        raise ValueError(msg)

class Compiler():
    def __init__(self, input_dir, save_dir, ext='nc'):
        self.input_dir = input_dir
        self.save_dir = save_dir
        self.ext = ext

        _check_input_filetype(self.ext)

        self.files = get_list_filenames(self.input_dir, self.ext)
        self.files = sorted(self.files)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load(self, filepath: str) -> np.ndarray:
        ds = xr.open_dataset(filepath)
        data = ds.Rad.values
        wavelengths = ds.band_wavelength.values.tolist()
        wavelengths = [np.round(w, 4) for w in wavelengths]
        del ds # free up memory
        return data, wavelengths

    def extract_datetime(self, filepath: str) -> str:
        datetime_str = filepath.split('/')[-1].split('_')[0]
        return datetime_str

    def mean_per_channel(self, data: np.ndarray) -> np.ndarray:
        mean = np.nanmean(data, axis=(1, 2))
        mean = [np.round(m, 6) for m in mean]
        return mean

    def std_per_channel(self, data: np.ndarray) -> np.ndarray:
        std = np.nanstd(data, axis=(1, 2))
        std = [np.round(s, 6) for s in std]
        return std
    
    def max_per_channel(self, data: np.ndarray) -> np.ndarray:
        max_ = np.nanmax(data, axis=(1, 2))
        max_ = [np.round(m, 6) for m in max_]
        return max_

    def min_per_channel(self, data: np.ndarray) -> np.ndarray:
        min_ = np.nanmin(data, axis=(1, 2))
        min_ = [np.round(m, 6) for m in min_]
        return min_

    def compile_metrics(self, files: list):
        
        means = []
        stds = []
        maxs = []
        mins = []
        datetimes = []
        wavelengths = []

        pbar = tqdm(files)
        for file in pbar:
            data, wvls = self.load(file)
            mean = self.mean_per_channel(data)
            std = self.std_per_channel(data)
            max_ = self.max_per_channel(data)
            min_ = self.min_per_channel(data)

            datetime_str = self.extract_datetime(file)

            means.append(mean)
            stds.append(std)
            maxs.append(max_)
            mins.append(min_)
            datetimes.append(datetime_str)
            wavelengths.append(wvls)


        df = pd.DataFrame({
            'datetime': datetimes,
            'wavelengths': wavelengths,
            'mean': means,
            'std': stds,
            'max': maxs,
            'min': mins,
        })
            
        file_0 = files[0].split('/')[-1].split('_')[0]
        file_n = files[-1].split('/')[-1].split('_')[0]
        filename = f'{file_0}-{file_n}'
        save_path = os.path.join(self.save_dir, f'{filename}.csv')
        df.to_csv(save_path, index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert files to a new format.')
    parser.add_argument('--input_dir', 
                        type=str, 
                        help='path to the files.')
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='path to save the converted files.')
    parser.add_argument('--ext',
                        default='nc',
                        type=str,
                        help='file extension of the input files.')
    args = parser.parse_args()

    logger.info("Initializing metrics calculation...")

    compiler = Compiler(args.input_dir, args.save_dir, args.ext)

    # chunk files
    cpus = int(multiprocessing.cpu_count())
    files = compiler.files
    chunk_files = np.array_split(files, cpus)
    chunk_list = [list(chunk) for chunk in chunk_files]

    logger.info(f"Converting {len(chunk_files)} chunks using {multiprocessing.cpu_count()} CPUs")

    with multiprocessing.Pool(cpus) as p:
        p.map(compiler.compile_metrics, chunk_list)
        p.close()
        p.join()

    logger.info("Done.")