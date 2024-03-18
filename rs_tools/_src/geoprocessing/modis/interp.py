import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# TODO: Satpy can interpolate grid, so function not necessarily needed
def interp_coords_modis(ds: xr.Dataset, desired_size: tuple[int], method: str="cubic") -> np.array:

    lat = ds.Latitude
    lon = ds.Longitude
    
    # Create grid coordinates for the original and interpolated arrays
    original_rows = np.stack((np.linspace(0, lat.shape[0], num=lat.shape[0]),) * lat.shape[1], axis=1)
    original_cols = np.stack((np.linspace(0, lat.shape[1], num=lat.shape[1]),) * lat.shape[0], axis=0)
    
    # Determine scale factors to match original and desired size
    sf_x = desired_size[0]/lat.shape[0] # scale factors do not need to be integers
    sf_y = desired_size[1]/lat.shape[1]
    original_rows = (original_rows) * sf_x
    original_cols = (original_cols) * sf_y
    interp_rows, interp_cols = np.indices(desired_size)

    # Flatten the original array and grid coordinates
    original_positions = np.column_stack((original_rows.flatten(), original_cols.flatten()))
    original_lat = lat.values.flatten()
    original_lon = lon.values.flatten()

    interp_positions = np.column_stack((interp_rows.flatten(), interp_cols.flatten()))

    # Perform 2D interpolation using griddata
    interp_lat = griddata(original_positions, original_lat, interp_positions, method=method)
    interp_lon = griddata(original_positions, original_lon, interp_positions, method=method)

    # Reshape the interpolated array to the desired size
    interp_lat = interp_lat.reshape(desired_size)
    interp_lon = interp_lon.reshape(desired_size)

    return interp_lat, interp_lon