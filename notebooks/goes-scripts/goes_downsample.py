from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import numpy as np
from xy_to_latlon import calc_latlon

def downsample(ds, target_points):
    """
    target_points
    """
    # attach latitude & longitude coordinates to xr.Datasets
    ds = calc_latlon(ds)
    
    # extract 1d arrays of latitudes and longitudes
    lat = ds.lat.to_numpy().flatten()
    lon = ds.lon.to_numpy().flatten()
    
    # turn into 2d array of latitudes and longitudes
    points = np.vstack((lon, lat)).T
    
    # initialise interpolation
    nn_interpolation = NearestNDInterpolator(points, ds.Rad)
    interpolated_nn = nn_interpolation(target_points)
    
    # create new xr.Dataset with lowres data 
    ...