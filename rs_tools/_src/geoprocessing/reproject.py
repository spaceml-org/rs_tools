from typing import Tuple
from pyproj import CRS, Transformer

def convert_lat_lon_to_x_y(crs: str, lon: list[float], lat: list[float]) -> Tuple[float, float]:
    transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y
