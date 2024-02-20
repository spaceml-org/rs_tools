#!/usr/bin/env python
"""utils_fidelity.py

This file contains utility functions called by the scripts in the ITI
fidelity testing pipeline.

Functions:
- csv_read_to_list -> List:
    Parse a CSV file into a list-of-lists.
- twodgaussian() -> Numpy array | function:
    Return a 2D Gaussian.
- moments() -> List:
    Calculate the moments of 2D data.
- vect_to_mpfit_parms() -> List:
    Convert a parameter vector to the MPFIT parameter structure.
- get_parm_vector() -> List:
    Extract a vector of parameter entries from the MPFIT parmeter structure.

"""

import os
import re
import sys
import numpy as np
import math


# -----------------------------------------------------------------------------#
def csv_read_to_list(file_name, delim=",", do_float=False):
    """
    Read rows from an ASCII file into a list of lists.
    """

    # Compile a few useful regular expressions
    spaces = re.compile("\s+")
    comma_and_spaces = re.compile(",\s+")
    comma_or_space = re.compile("[\s|,]")
    brackets = re.compile("[\[|\]\(|\)|\{|\}]")
    comment = re.compile("#.*")
    quotes = re.compile("'[^']*'")
    keyVal = re.compile("^.+=.+")
    words = re.compile("\S+")

    # Parse the input file into a list
    out_lst = []
    FH = open(file_name, "r")
    for line in FH:
        line = line.rstrip("\n\r")
        if comment.match(line):
            continue
        line = comment.sub("", line)  # remove internal comments
        line = line.strip()           # kill external whitespace
        line = spaces.sub(" ", line)  # shrink internal whitespace
        if line == "":
            continue
        line = line.split(delim)
        if len(line) < 1:
            continue
        if do_float:
            line = [float(x) for x in line]

        out_lst.append(line)

    return out_lst


#-----------------------------------------------------------------------------#
def twodgaussian(params, shape=None):
    """
    Function to build a 2D Gaussian ellipse as parameterised by 'params':

        params - [amp, x0, y0, sig_x, sig_y, pa] where:
                  amp    - amplitude
                  x0     - centre of Gaussian in X
                  y0     - centre of Gaussian in Y
                  sig_x  - width of Gaussian in X (sigma or c, not FWHM)
                  sig_y  - width of Gaussian in Y (sigma or c, not FWHM)
                  pa     - position angle of Gaussian (degrees)
        shape - (y, x) dimensions of region (optional)

    If called without a shape, returns a function with the parameters
    'baked in' that can be used by a fitter. If called with a shape, it
    evaluates the function and returns a Numpy array.

    """

    amp, x0, y0, sig_x, sig_y, pa = params
    pa %= 180.0
    pa = np.radians(pa)

    def gauss(y, x):
        st = np.sin(pa)**2
        ct = np.cos(pa)**2
        s2t = np.sin(2*pa)
        a = (ct/sig_x**2 + st/sig_y**2)/2
        b = s2t/4 *(1/sig_y**2-1/sig_x**2)
        c = (st/sig_x**2 + ct/sig_y**2)/2
        v = amp * np.exp(-1*(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))
        return v

    if shape is not None:
        return gauss(*np.indices(shape))
    else:
        return gauss


#-----------------------------------------------------------------------------#
def moments(data):
    """
    Calculate the moments of 2D data.

    Returns: height, x, y, width_x, width_y, pa
    """

    total = data.sum()
    height = data.max()
    xi, yi = np.indices(data.shape)
    x = (xi * data).sum() / total
    y = (yi * data).sum() / total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    pa = 0.0

    return height, x, y, width_x, width_y, pa


#-----------------------------------------------------------------------------#
def vect_to_mpfit_parms(p, p_names=None, p_fixed=None, p_limits=None):
    """
    Convert a vector of parameter values to a MPFIT parameter structure.
    The structure is a list of dictionaries with this format:

      parms=[ {'value': 5.1,
               'fixed': False,
               'parname': 'amp',
               'limited': [False, False]}, ... ]

    The 'fixed' keyword freezes the variable at the provided value,
    the 'parname'keyword allows the provision of a name (including
    LaTex code), and the 'limited' keyword supports the setting of
    fitting bounds.

    """

    mpfit_parm_lst = []
    for idx, value in enumerate(p):
        mpfit_parm_lst.append({'value': value,
                               'fixed': False,
                               'parname': f"Var_{idx}",
                               'limited': [False, False]})
    if p_names is not None:
        if len(p_names) == len(p):
            for idx, value in enumerate(p_names):
                mpfit_parm_lst[idx]["parname"] = value
    if p_fixed is not None:
        if len(p_fixed) == len(p):
            for idx, value in enumerate(p_fixed):
                mpfit_parm_lst[idx]["fixed"] = bool(value)
    if p_limits is not None:
        if len(p_limits) == len(p):
            for idx, value in enumerate(p_names):
                mpfit_parm_lst[idx]["limited"] = value

    return mpfit_parm_lst


#-----------------------------------------------------------------------------#
def get_parm_vector(parms, field_name="value"):
    """
    Get a vector of parameters given a field name.
    Allowed field names are ['value', 'fixed', 'parname', 'limited'].
    """

    if not field_name in ['value', 'fixed', 'parname', 'limited']:
        return [None] * len(parms)

    val_lst = []
    for idx, e in enumerate(parms):
        val_lst.append(e[field_name])

    return val_lst
