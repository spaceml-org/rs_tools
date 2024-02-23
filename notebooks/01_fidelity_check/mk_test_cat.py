#!/usr/bin/env python
"""
mk_test_cat.py

This script generates an ASCII catalogue file with features that cover
the extent of the test image.

Functions:
- gen_catalogue_file() -> None:
    Generate an ASCII catalogue of features.
- plot_gaussian_cat() -> None:
    Plot the Generated Gaussians as FWHM ellipses on a figure.

"""

import os
import re
import sys
import math
import csv
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection
from tqdm import tqdm


#-----------------------------------------------------------------------------#
def gen_catalogue_file(out_root, num_features, jitter=0.0,
                       amp_range_gauss=None, width_range_gauss=[20.0, 50.0],
                       aspect_max_gauss=3.0, pa_range_gauss=[0.0, 180.0],
                       size=2000, debug=False):
    """
    Generate an ASCII file containing features that spread across an image.
    """

    # Lay out the grid of initial x-y positions assuming a square image
    n_lines = np.ceil(num_features**0.5).astype(int)
    n_places = n_lines ** 2
    print(f"[INFO] # features set at {n_places} to cover square grid.")
    dxy = 1 / n_lines
    x_lst = []
    y_lst = []
    for i_ in range(n_lines):
        x = dxy / 2 + dxy * i_
        for j_ in range(n_lines):
            y = dxy / 2 + dxy * j_
            x_lst.append(x)
            y_lst.append(y)
    x_arr = np.array(x_lst, dtype="f4")
    y_arr = np.array(y_lst, dtype="f4")

    # Jitter the positions
    pos_shift = np.random.uniform(low=-jitter, high=jitter, size=n_places)
    x_arr += pos_shift
    pos_shift = np.random.uniform(low=-jitter, high=jitter, size=n_places)
    y_arr += pos_shift

    # Determine the numbers of each type of feature
    # TODO: Write after adding more feature types

    # Generate the range of Gaussian parameters:
    # Amplitude
    amp_arr = np.random.uniform(low=amp_range_gauss[0],
                                high=amp_range_gauss[1],
                                size=n_places)
    # Width X (before rotation)
    sig_x_arr = np.random.uniform(low=width_range_gauss[0],
                                  high=width_range_gauss[1],
                                  size=n_places)
    # Width Y (before rotation)
    aspect_arr = np.random.uniform(low=1.0/aspect_max_gauss,
                                   high=1.0,
                                   size=n_places)
    sig_y_arr = sig_x_arr * aspect_arr
    # PA
    pa_arr = np.random.uniform(low=pa_range_gauss[0],
                               high=pa_range_gauss[1],
                               size=n_places) % 180.0

    # Write out the catalogue file, including preamble
    cat_filename = out_root + ".csv"
    if os.path.exists(cat_filename):
        input(f"[WARN] File '{cat_filename}' exists!\n"
              f"Press <RETURN> to overwrite ...")
    print(f"[INFO] Writing {n_places} features to '{cat_filename}' ...", end="")
    with open(cat_filename, "w") as FH:
        FH.write(f"#" + "-" * 77 + "#\n"
                 f"# Catalogue for use with ITI validation workflow.\n"
                 f"#" + "-" * 77 + "#\n\n"
                 f"# Type 1 = Gaussian Features\n# type, amp, x_frac, y_frac, "
                 f"sig_x_pix, sig_y_pix, pa_deg\n\n")
        w = csv.writer(FH, delimiter=",", quotechar="'",
                       quoting=csv.QUOTE_MINIMAL)
        for i_ in range(n_places):
            w.writerow([1,
                        amp_arr[i_],
                        x_arr[i_],
                        y_arr[i_],
                        sig_x_arr[i_],
                        sig_y_arr[i_],
                        pa_arr[i_]])
    print(" done.")

    # Plot the positions
    if debug:
        fig = plt.figure(figsize=(6.5, 6))
        plot_gaussian_cat(fig, size, amp_arr, x_arr, y_arr,
                          sig_x_arr, sig_y_arr, pa_arr)
        fig.canvas.draw()
        fig_filename = out_root + ".png"
        fig.savefig(fig_filename, dpi=300)
        fig.show()
        input("Press <RETURN> to continue ...")


#-----------------------------------------------------------------------------#
def plot_gaussian_cat(fig, size, amp_arr, x_arr, y_arr, sig_x_arr, sig_y_arr,
                      pa_arr):
    """
    Plot the Generated Gaussians as FWHM ellipses on a figure.
    """

    # Setup axes for plotting
    ax1 = fig.add_subplot(1,1,1)

    # Plot the Gaussians as full-width half-max (FWHM) ellipses
    sigma2fwhm = math.sqrt(8*math.log(2))
    ellipse_lst = []
    for i_ in range(len(x_arr)):
        el = Ellipse(xy=(x_arr[i_], y_arr[i_]),
                     width=sig_x_arr[i_] * sigma2fwhm / size,
                     height=sig_y_arr[i_] * sigma2fwhm / size,
                     angle=-1*pa_arr[i_])
        ellipse_lst.append(el)
    ellipses = PatchCollection(ellipse_lst, cmap='plasma', alpha=0.7,
                               edgecolor="k", lw=1)
    ellipses.set_array(amp_arr)
    ax1.add_collection(ellipses)

    # Format figure and add a colourbar
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (frac)")
    ax1.set_ylabel("Y (frac)")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.00)
    cbar1 = plt.colorbar(ellipses, cax=cax, spacing="proportional",
                         orientation="vertical")
    cbar1.set_label("Amplitude")


#-----------------------------------------------------------------------------#
if __name__ == '__main__':

    desc_str = """
    Create a new catalogue of features to populate a testing
    image. This initial version generates Gaussian features with
    uniformly distributed properties between two limits.

    Usage:

        python mk_test_cat.py \\
            --out testcat \\
            --amp-range-gauss 20 50 \\
            --size 2000 \\
            --debug

    This example generates a CSV file called 'testcat.csv' with each
    row describing the parameters of a Gaussian feature:

        amplitude (image units),
        position_x (fractional),
        position_y (fractional),
        sigma_x (pixels),
        sigma_y (pixels),
        position_angle (degrees)

    The file can be used by the 'mk_test_image.py' script to generate
    a validation image populated with Gaussians.
    """

    epilog_str = """
    Copyleft Trillium Technologies 2024.
    Queries to team@trillium.tech.
    """

    # Parse the command line options
    ap = argparse.ArgumentParser(description=desc_str, epilog=epilog_str,
                                 formatter_class=argparse.RawTextHelpFormatter)

    ap.add_argument(
        "--out",
        required=True,
        help="Root name of the output catalogue file."
    )
    ap.add_argument(
        "--num-features",
        default=9,
        type=int,
        help=(f"Number of features to generate [%(default)s]. The actual\n"
              f"number generated will be expanded to fill the image.")
    )
    ap.add_argument(
        "--jitter",
        default=0.01,
        type=float,
        help="Randomise x-y position by fractional amount [%(default)s]."
    )
    gauss_grp = ap.add_argument_group('gaussians')
    gauss_grp.add_argument(
        "--amp-range-gauss",
        nargs=2,
        type=float,
        help="Range of amplitudes for the Gaussian features."
    )
    gauss_grp.add_argument(
        "--width-range-gauss",
        nargs=2,
        default=[20.0, 50.0],
        type=float,
        help="Range of sigma widths (before rotation) [%(default)s]."
    )
    gauss_grp.add_argument(
        "--aspect-max-gauss",
        default=3.0,
        type=float,
        help="Maximum aspect-ratio for the Gaussians [%(default)s]."
    )
    gauss_grp.add_argument(
        "--pa-range-gauss",
        nargs=2,
        default=[0.0, 180.0],
        type=float,
        help="Position angle range for the Gaussians [%(default)s]."
    )
    ap.add_argument(
        "--size",
        default=2000,
        type=int,
        help="Number of pixels on X and Y axes (equal) [%(default)s]."
    )
    gauss_grp.add_argument(
        "--num-weight-gauss",
        default=1.0,
        type=float,
        help="Relative weight to control # Gaussians created [%(default)s]."
    )
    ap.add_argument(
        "--debug",
        action='store_true',
        help="Show debugging messages and plots."
    )
    args = ap.parse_args()

    # Check Gaussian args are present and complete
    gauss_features_OK = True
    if (args.amp_range_gauss is None
        or args.width_range_gauss is None
        or args.aspect_max_gauss is None):
        gauss_features_OK = False
    if gauss_features_OK:
        args.amp_range_gauss.sort()
        args.width_range_gauss.sort()

    # Check at least one feature group is specified
    if sum([gauss_features_OK]) == 0:
        sys.exit("[ERR] At least one feature group must be specified.")

    # Call the funtion to generate the catalogue
    gen_catalogue_file(
        out_root=args.out,
        num_features=args.num_features,
        jitter=args.jitter,
        amp_range_gauss=args.amp_range_gauss,
        width_range_gauss=args.width_range_gauss,
        aspect_max_gauss=args.aspect_max_gauss,
        pa_range_gauss=args.pa_range_gauss,
        size=args.size,
        debug=args.debug)
