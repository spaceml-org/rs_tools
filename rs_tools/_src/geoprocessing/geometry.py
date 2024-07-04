import rasterio
from georeader.window_utils import window_polygon, polygon_to_crs, compare_crs


def calculate_xrio_footprint(da, dst_crs=None):

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