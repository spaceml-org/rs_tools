import rioxarray
import xarray as xr
from goes2go import GOES # activate the rio accessor


def correct_goes16_satheight(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert measurement angle of GOES-16 satellite data to horizontal distance (in meters).

    Parameters:
    - ds (xr.Dataset): The input dataset containing the GOES-16 satellite data.

    Returns:
    - xr.Dataset: The dataset with corrected perspective height.
    """
    # reassign coordinates to correct height
    x_attrs = ds.x.attrs
    ds = ds.assign_coords({"x": ds.FOV.x})
    ds["x"].attrs = x_attrs
    ds["x"].attrs["units"] = "meters"

    y_attrs = ds.y.attrs
    ds = ds.assign_coords({"y": ds.FOV.y})
    ds["y"].attrs = y_attrs
    ds["y"].attrs["units"] = "meters"

    return ds


def correct_goes16_bands(ds: xr.Dataset) -> xr.Dataset:
    """
    Corrects the band coordinates in a GOES-16 dataset.

    Parameters:
        ds (xr.Dataset): The input dataset containing GOES-16 bands.

    Returns:
        xr.Dataset: The corrected dataset with updated band coordinates.

    """
    # reassign coordinate
    band_id_attrs = ds.band_id.attrs
    ds = ds.assign_coords(band=ds.band_id.values)
    ds.band.attrs = band_id_attrs

    # drop bandid dims
    ds = ds.drop_vars("band_id")
    return ds
