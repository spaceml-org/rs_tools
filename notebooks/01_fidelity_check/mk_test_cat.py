#!/usr/bin/env python
"""
mk_test_cat.py

This script generates an ASCII catalogue file with features that cover
the extent of the test image.

Functions:
- gen_catalogue_file() -> None:
    Generate a catalogue of features.

"""

import os
import re
import sys
import argparse
from tqdm import tqdm


#-----------------------------------------------------------------------------#
def gen_catalogue_file(out_root, num_features, jitter=0.0, amp_range_gauss=None,
                       maj_range_gauss=None, aspect_max_gauss=None,
                       pa_range_gauss=None):
    """
    Generate an ASCII file containing features that spread across an image.
    """

    # Sanity checks on the arguments

    # Lay out the grid of initial x-y positions

    # Jitter the positions

    # Determine the numbers of each type of feature



    # Generate the range of Gaussian parameters:
    # Amplitude

    # Maj-axis

    # Min-axis

    # PA


    # Write out the catalogue file, including preamble



#-----------------------------------------------------------------------------#
if __name__ == '__main__':

    desc_str = """
    Create a new catalogue of features to populate a testing
    image. This initial version generates Gaussian features with
    uniformly distributed properties between two limits."""

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
        default=50,
        type=int,
        help="Number of features to generate [%(default)s]."
    )
    ap.add_argument(
        "--jitter",
        default=0.0,
        type=float,
        help="Randomise x-y position by [%(default)s] pixels."
    )
    gauss_grp = ap.add_argument_group('gaussians')
    gauss_grp.add_argument(
        "--amp-range-gauss",
        nargs=2,
        type=float,
        help="Range of amplitudes for the Gaussian features."
    )
    gauss_grp.add_argument(
        "--maj-range-gauss",
        nargs=2,
        default=[20.0, 50.0],
        type=float,
        help="Range of maj-axis values for the Gaussian features [%(default)s]."
    )
    gauss_grp.add_argument(
        "--aspect-max-gauss",
        default=2.0,
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
    gauss_grp.add_argument(
        "--num-weight-gauss",
        default=1.0,
        type=float,
        help="Relative weight to control # Gaussians created [%(default)s]."
    )
    args = ap.parse_args()

    # Check Gaussian args are present and complete
    gauss_features_OK = True
    if (args.amp_range_gauss is None
        or args.maj_range_gauss is None
        or args.aspect_max_gauss is None):
        gauss_features_OK = False
    if gauss_features_OK:
        args.amp_range_gauss.sort()
        args.maj_range_gauss.sort()

    # Check at least one feature group is specified
    if sum([gauss_features_OK]) == 0:
        sys.exit("[ERR] At least one feature group must be specified.")

    # Call the funtion to generate the catalogue
    gen_catalogue_file(
        out_root=args.out,
        num_features=args.num_features,
        jitter=args.jitter,
        amp_range_gauss=args.amp_range_gauss,
        maj_range_gauss=args.maj_range_gauss,
        aspect_max_gauss=args.aspect_max_gauss,
        pa_range_gauss=args.pa_range_gauss)
