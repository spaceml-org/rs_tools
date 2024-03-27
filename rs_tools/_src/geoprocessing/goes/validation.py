import rioxarray
import xarray as xr


def correct_goes16_satheight(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert measurement angle of GOES-16 satellite data to horizontal distance (in meters).

    Parameters:
    - ds (xr.Dataset): The input dataset containing the GOES-16 satellite data.

    Returns:
    - xr.Dataset: The dataset with corrected perspective height.
    """

    # get perspective height
    sat_height = ds.goes_imager_projection.attrs["perspective_point_height"]

    # reassign coordinates to correct height
    x_attrs = ds.x.attrs
    ds = ds.assign_coords({"x": ds.x.values * sat_height})
    ds["x"].attrs = x_attrs
    ds["x"].attrs["units"] = "meters"

    y_attrs = ds.y.attrs
    ds = ds.assign_coords({"y": ds.y.values * sat_height})
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
