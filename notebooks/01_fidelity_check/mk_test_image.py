#!/usr/bin/env python
"""
mk_test_image.py

This script provides functionality for generating fidelity testing
images from an ASCII-format catalogue of features.

Functions:
- gen_testing_file() -> None:
    Generate the test image given a catalogue.
- csv_read_to_list -> List:
    Parse a CSV file into a list-of-lists.
- def twodgaussian() -> Numpy array:
    Generate a 2D Gaussian on a plane of zeros.

"""

import os
import re
import sys
import argparse
import numpy as np
import math
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm

from utils_fidelity import csv_read_to_list, twodgaussian


#-----------------------------------------------------------------------------#
def gen_testing_data(cat_file, out_root, noise_level, x_size, y_size, pixscale,
                     x_cent=284925.0, y_cent=6174486.0, crs="EPSG:32756"):
    """
    Generate a GeoTIFF file containing test structures from a catalogue file.
    Default to projection UTM zone 56S (EPSG:32756) near the city of
    Sydney, Australia.
    """

    # Read catalogue file
    if not os.path.exists(cat_file):
        sys.exit(f"[ERR] File does not exist: '{cat_file}'.")
    cat_lst = csv_read_to_list(cat_file, do_float=True)
    num_entries = len(cat_lst)
    print(f"[INFO] found {num_entries} entries in the catalogue.")

    # Create a blank array for the GeoTIFF
    data_arr = np.zeros(shape=(y_size, x_size), dtype="f4")

    # Loop through the catalogue and add the features
    for idx, e in enumerate(tqdm(cat_lst, desc="GENERATING FEATURES")):
        src_type = int(e[0])

        # Feature Type 1: Gaussian features.
        # twodgaussian() expects [amp, xo, yo, cx, cy, pa]
        if src_type == 1:
            xo = e[1] * x_size
            yo = e[2] * y_size
            cx = e[3]
            cy = e[4]
            pa = e[5]
            amp = e[6]
            params = [amp, xo, yo, cx, cy, pa]
            shape2D = (y_size, x_size)
            gauss = twodgaussian(params, shape2D).reshape((y_size, x_size))
            data_arr[:, :] += gauss

    # Add the noise
    print(f"[INFO] Adding noise at scale: {noise_level}.")
    noise_arr = np.random.normal(scale=noise_level, size=(y_size, x_size))
    data_arr[:, :] += noise_arr

    # Calculate the image coordinate boundaries and CRS variables
    # See https://pygis.io/docs/d_raster_crs_intro.html
    x_min = x_cent - pixscale * x_size / 2.0
    y_max = y_cent + pixscale * y_size / 2.0
    tfm = Affine.translation(x_min, y_max) * Affine.scale(pixscale, pixscale)

    # Write a GeoTIFF to the current directory
    out_file = out_root + ".tif"
    print(f"[INFO] Writing to '{out_file}' ... ", end="", flush=True)
    with rasterio.open(
            fp=out_file,
            mode='w',
            driver='GTiff',
            height=y_size,
            width=x_size,
            count=1,
            dtype=data_arr.dtype,
            crs=crs,
        transform=tfm,
    ) as dst:
        dst.write(data_arr, 1)
    print("done.\n")


#-----------------------------------------------------------------------------#
if __name__ == '__main__':

    desc_str = """
    Create a new fidelity testing image from catalogues. This initial
    version understands catalogues containing Gaussian features on a
    background of uniform noise.
    """

    epilog_str = """
    Copyleft Trillium Technologies 2024.
    Queries to team@trillium.tech.
    """

    # Parse the command line options
    ap = argparse.ArgumentParser(description=desc_str, epilog=epilog_str,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--cat",
        required=True,
        help="Input catalogue file in CSV format."
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Root name of the output file(s)."
    )
    ap.add_argument(
        "--noise-level",
        default=10.0,
        type=float,
        help="Level of noise in the image [%(default)s]."
    )
    ap.add_argument(
        "--size",
        default=10000,
        type=int,
        help="Number of pixels on X and Y axes (equal) [%(default)s]."
    )
    ap.add_argument(
        "--pixscale",
        default=10.0,
        type=float,
        help="Scale of square pixels in projection units [%(default)s]."
    )
    args = ap.parse_args()

    # Call the function to generate the test data
    gen_testing_data(
        cat_file=args.cat,
        out_root=args.out,
        noise_level=args.noise_level,
        x_size=args.size,
        y_size=args.size,
        pixscale=args.pixscale)
