import os
import multiprocessing
import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

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

def _check_output_filetype(file_type: str) -> bool:
    """checks allowed output file types."""
    if file_type in ["npy", "npz"]:
        return True
    else:
        msg = "Unrecognized file type"
        msg += f"\nNeeds to be 'npy' or 'npz'. Others are not yet implemeneted."
        raise ValueError(msg)


class Converter():
    def __init__(self, input_dir, save_dir, ext='nc', dest='npy', prec = 'float32'):
        self.input_dir = input_dir
        self.save_dir = save_dir
        self.ext = ext
        self.dest = dest
        self.prec = prec

        _check_input_filetype(self.ext)
        _check_output_filetype(self.dest)

        self.files = get_list_filenames(self.input_dir, self.ext)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load(self, filepath):
        ds = xr.open_dataset(filepath)
        return ds

    def stack(self, ds):
        data = ds.Rad.values
        lats = ds.latitude.values
        lats = np.expand_dims(lats, axis=0)
        lons = ds.longitude.values
        lons = np.expand_dims(lons, axis=0)
        cloud_mask = ds.cloud_mask.values
        cloud_mask = np.expand_dims(cloud_mask, axis=0)

        stack = np.concatenate([data, cloud_mask, lats, lons], axis=0)
        stack = stack.astype(self.prec)
        del data, lats, lons, cloud_mask
        return stack

    def convert_files(self, files):
        pbar = tqdm(files)
        for file in pbar:
            self.convert(file)

    def convert(self, file):
        filename = file.split('/')[-1].split('.')[0]
        ds = self.load(file)
        stack = self.stack(ds)
        if self.dest == 'npy':
            np.save(os.path.join(self.save_dir, f"{filename}.{self.dest}"), stack)
        elif self.dest == 'npz':
            np.savez(os.path.join(self.save_dir, f"{filename}.{self.dest}"), stack)
        logger.info(f"Converted {filename}.{self.ext} to {filename}.{self.dest}")
        del ds, stack


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert files to a new format.')
    parser.add_argument('--input_dir', 
                        default = '/home/anna.jungbluth/data/geoprocessed/msg/',
                        type=str, 
                        help='path to the files.')
    parser.add_argument('--save_dir', 
                        default = '/home/anna.jungbluth/data/converted/msg/',
                        type=str, 
                        help='path to save the converted files.')
    parser.add_argument('--ext',
                        default='nc',
                        type=str,
                        help='file extension of the input files.')
    parser.add_argument('--dest',
                        default='npz',
                        type=str,
                        help='file extension of the converted files.')
    parser.add_argument('--prec',
                        default='float32',
                        type=str,
                        help='precision of the converted files.')
    args = parser.parse_args()

    logger.info("Initializing converter...")

    converter = Converter(args.input_dir, args.save_dir, args.ext, args.dest, args.prec)

    # chunk files
    cpus = int(multiprocessing.cpu_count())
    files = converter.files
    chunk_files = np.array_split(files, cpus)
    chunk_list = [list(chunk) for chunk in chunk_files]

    logger.info(f"Converting {len(chunk_files)} files using {multiprocessing.cpu_count()} CPUs")

    with multiprocessing.Pool(cpus) as p:
        p.map(converter.convert_files, chunk_list)
        p.close()
        p.join()

    logger.info("Done.")