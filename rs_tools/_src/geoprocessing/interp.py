# populates the search tree
import xarray as xr
import pyinterp
import numpy as np
from scipy.interpolate import interp1d
from rs_tools._src.utils.math import bounds_and_step_to_points


SCALE_NUM_PTS = {
    "1": 10_848, # Bands - 1,3,5
    "0.5": 21_696, # Bands - 2
    "2": 5_424, # Bands - 4,6,7,8,9,10,11,12,13,14,15,16
}

# FIXED COORDS FOR GOES 16 (ONLY?)
X0, X1 = - 0.15185800194740295, 0.15185800221843238

DX_GOES16 = [
    1.3999354882038965e-05,
    2.8000000384054147e-05,
    5.600516396198328e-05,
]

RES_GOES16 = [
    0.5,
    1,
    2,
]


def create_goes16_interp_mesh(ds: xr.Dataset, variable: str="Rad"):

    mesh = pyinterp.RTree()

    # extract coordinates - 2D vector
    X, Y = np.meshgrid(ds.x.values, ds.y.values)

    # ravel them - 1D vectors
    X, Y = X.ravel(), Y.ravel()

    mesh.packing(
        np.vstack((X,Y )).T,
        ds[variable].values.ravel()
    )
    return mesh


def create_goes16_coords(scale: float=1.0):

    DX = interp1d(RES_GOES16, DX_GOES16, kind='linear', bounds_error=False, fill_value="extrapolate")(scale)
    NX = bounds_and_step_to_points(X0, X1, DX)

    # create coordinate vector
    x_coords = np.linspace(X0, X1, NX)

    return x_coords


def resample_goes16(ds, scale: float=1.0):

    # create interpolation mesh
    # print("Creating interp mesh...")
    mesh = create_goes16_interp_mesh(ds, variable="Rad")

    # create query coordinates
    # print("Creating coords...")
    x_coords = create_goes16_coords(scale=scale)    

    # create meshgrid
    # print("Creating meshgrid")
    X, Y = np.meshgrid(x_coords, x_coords)

    # Inverse Distance Weighting
    # print("doing distance calc...")
    idw_eta, neighbors = mesh.inverse_distance_weighting(
        np.vstack((X.ravel(), Y.ravel())).T,
        within=True,  # Extrapolation is forbidden
        radius=5500,  # In a radius of 5.5 Km
        k=8,  # We are looking for at most 8 neighbours
        num_threads=0, # Parallel processing
    )
    idw_eta = idw_eta.reshape(X.shape)

    ds_new = xr.Dataset(
        {"Rad": (("band","y", "x"), idw_eta[None, ...])},
        coords = {"x": (("x"), x_coords),
                  "y": (("y"), x_coords),
                  "band": ds.band_id.values}
    )

    
    ds_new.Rad.attrs = ds.Rad.attrs
    dx = interp1d(RES_GOES16, DX_GOES16, kind='linear', bounds_error=False, fill_value="extrapolate")(scale)
    ds_new.Rad.attrs["resolution"] = f"y: {dx} rad x: {dx} rad"
    ds_new.Rad.attrs["resolution_km"] = f"y: {scale} km x: {scale} km"
    ds_new.attrs = ds.attrs
    ds_new["goes_imager_projection"] = ds["goes_imager_projection"]

    return ds_new