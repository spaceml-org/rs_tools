import rasterio
from georeader.window_utils import window_polygon, polygon_to_crs, compare_crs
from odc.geo.geom import BoundingBox
from shapely.geometry import box


def bbox_string_to_bbox(
    bbox_string: str="-180 -90 180 90",
):
    """
    Convert a string representation of a bounding box to a BoundingBox object.

    Args:
        bbox_string (str): The string representation of the bounding box in the format "lon_min lat_min lon_max lat_max".

    Returns:
        BoundingBox: The BoundingBox object representing the bounding box.

    """
    lon_min, lat_min, lon_max, lat_max = bbox_string.split(" ")
    bbox = BoundingBox.from_xy(x=(float(lon_min), float(lon_max)), y=(float(lat_max), float(lat_min)), crs="4326")
    return bbox


def calculate_xrio_footprint_v2(da):
    return box(*da.rio.bounds())


def calculate_xrio_footprint_reproject(da, dst_crs):
    return box(*da.rio.transform_bounds(dst_crs))


def calculate_xrio_footprint(da, dst_crs=None):
    """
    Calculate the footprint of a rasterio xarray object.

    Parameters:
    - da (xarray.DataArray): The input xarray DataArray object representing the raster.
    - dst_crs (str or CRS, optional): The destination coordinate reference system (CRS) to reproject the footprint to. 
      If not provided, the function returns the footprint in the same CRS as the input raster.

    Returns:
    - polygon (shapely.geometry.Polygon): The footprint of the raster as a Shapely Polygon object.

    Note:
    - The function uses the rioxarray library to extract the window, transform, and CRS information from the input raster.
    - If the destination CRS is provided and different from the source CRS, the function reprojects the footprint to the destination CRS using the `polygon_to_crs` function.

    """

    # get window from rioxarray
    window = rasterio.windows.Window(
        row_off=0, col_off=0, 
        height=da.rio.height, 
        width=da.rio.width
    )

    # get transform
    transform = da.rio.transform()

    # get coordinate reference system
    src_crs = da.rio.crs

    # calculate polygon from window
    polygon = window_polygon(window, transform)

    if (dst_crs is None) or compare_crs(src_crs, dst_crs):
        return polygon
    
    return polygon_to_crs(polygon, src_crs, dst_crs)