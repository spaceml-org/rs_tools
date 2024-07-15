import xarray as xr


def add_crs_from_rio(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds the Coordinate Reference System (CRS) to the given xarray Dataset using the rio library.

    Parameters:
        ds (xr.Dataset): The xarray Dataset to which the CRS will be added.

    Returns:
        xr.Dataset: The updated xarray Dataset with the CRS added.
    """

    # load CRS variable
    crs_variable = ds["spatial_ref"].attrs["crs_wkt"]

    # get attribute with wkt
    
    # assign CRS to dataarray
    return ds.rio.write_crs(crs_variable)